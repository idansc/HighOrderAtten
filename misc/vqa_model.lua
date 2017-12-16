

require 'nngraph'
require 'nn'
require 'rnn'
require 'cunn'
require 'misc.maskSoftmax'
require 'misc.CompactBilinearPooling'
require 'cbp.SignedSquareRoot'
local utils = require 'misc.utils'
local attention = require 'misc.attention'
--nngraph.setDebug(true)

local layer, parent = torch.class('nn.vqa_model', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.seq_length = utils.getopt(opt, 'seq_length')
	self.feature_type = utils.getopt(opt, 'feature_type')
	self.dropout = utils.getopt(opt, 'dropout', 0)
	--embeddings dims
	self.hidden_size = utils.getopt(opt, 'hidden_size')
	self.hidden_last_size = utils.getopt(opt, 'hidden_last_size')
    self.hidden_combine_size = utils.getopt(opt, 'hidden_combine_size')
	self.output_size = utils.getopt(opt, 'output_size')
	

	
	if self.feature_type == 'VGG' then
		self.cnn = nn.Sequential()
                      :add(nn.View(512):setNumInputDims(2))
                      :add(nn.Linear(512, self.hidden_size))					 
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
					  :add(nn.Dropout(dropout))

    elseif self.feature_type == 'Residual' then
        self.cnn = nn.Sequential()
                      :add(nn.View(2048):setNumInputDims(2))
                      :add(nn.Linear(2048, self.hidden_size))
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
					  :add(nn.Dropout(dropout))

    end
    self.mask = torch.ByteTensor()
    self.atten = attention.margin_attention(self.seq_length, 196, 18, self.hidden_size)	 
    self.vqa = vqa(self.atten,self.cnn,self.vocab_size, self.seq_length, self.hidden_size, self.hidden_last_size, self.hidden_combine_size, self.output_size, self.dropout)
end

function layer:getModulesList()
    return {self.vqa, self.cnn}
end

function layer:parameters()
    
	local p1,g1 = self.cnn:parameters()
    local p2,g2 = self.vqa:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p2) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.vqa:training()
    self.cnn:training()
end

function layer:evaluate()
    self.vqa:evaluate()
    self.cnn:evaluate()
end

function layer:updateOutput(input)
  local ques = input[1]
  local img = input[2]
  local mc_ans = input[3]
  self.mask:resize(ques:size()):zero()
  self.mask[torch.eq(ques, 0)] = 1
  self.output = self.vqa:forward({ques,img,self.mask,mc_ans})
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local img = input[2]
  local mc_ans = input[3]
  
  local dummy = unpack(self.vqa:backward({ques, img,self.mask, mc_ans}, gradOutput))

  return self.gradInput
end



function vqa(atten,cnn,vocab_size,seq_length,ques_embedding_size, hidden_last_size, hidden_combine_size ,output_size, dropout)
	local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
	
    local question = inputs[1]
	local img = inputs[2]
	local mask = inputs[3]
    local MC_ans = inputs[4]
    
	img_feat =cnn(img)
	-- Word Embedding
    local embed = nn.Dropout(0.5)(nn.Tanh()(nn.LookupTableMaskZero(vocab_size, ques_embedding_size)(question)))
	
	-- 1D Conv for bigrams and trigrams
	

	local trigram = nn.Tanh()(cudnn.TemporalConvolution(ques_embedding_size, ques_embedding_size,3, 1, 1)(embed))

	-- LSTM
	
	local rnn1 = nn.SeqLSTM(ques_embedding_size,ques_embedding_size/2)
	rnn1.maskzero = true
	rnn1.batchfirst = true
	rnn1.usenngraph = true
	local rnn2 = nn.SeqLSTM(ques_embedding_size, ques_embedding_size/2)
	rnn2.maskzero = true
	rnn2.batchfirst = true
	rnn2.usenngraph = true

	local seq_feat = nn.Dropout(0.5)(nn.JoinTable(2,2){
			rnn1(embed),
			rnn2(trigram)
			})
	  
	
    --MC_embed 
    local mc_embed = nn.Dropout(0.5)(nn.Tanh()(nn.LookupTableMaskZero(output_size, ques_embedding_size)(MC_ans)))
	-- Attention
	local attented_ques, attented_img, attented_ans = atten({seq_feat,img_feat,mc_embed,mask}):split(3)

	--MCB + FC
   
	local combine_ans_img = nn.Normalize(2)(nn.SignedSquareRoot()(nn.CompactBilinearPooling(hidden_last_size,tostring(ques_embedding_size) .. hidden_last_size)({attented_ans, attented_img})))
	local combine_ques_img = nn.Normalize(2)(nn.SignedSquareRoot()(nn.CompactBilinearPooling(hidden_last_size,tostring(ques_embedding_size) .. hidden_last_size)({attented_ques, attented_img})))
    local last_fc_hidden = nn.Normalize(2)(nn.SignedSquareRoot()(nn.CompactBilinearPooling(hidden_combine_size,tostring(hidden_combine_size) .. hidden_last_size)({combine_ans_img, combine_ques_img})))
    local outfeat = nn.Linear(hidden_combine_size, output_size)(nn.Dropout(0.3)(last_fc_hidden))


    --nngraph.annotateNodes()

    table.insert(outputs, outfeat)
    
    return nn.gModule(inputs, outputs)
end