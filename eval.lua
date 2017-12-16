------------------------------------------------------------------------------
--  Hierarchical Question-Image Co-Attention for Visual Question Answering
--  J. Lu, J. Yang, D. Batra, and D. Parikh
--  https://arxiv.org/abs/1606.00061, 2016
--  if you have any question about the code, please contact jiasenlu@vt.edu
-----------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'optim'
require 'gnuplot'
require 'misc.DataLoaderDisk'
require 'misc.vqa_model'
local utils = require 'misc.utils'
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_test_h5','/home/idansc@st.technion.ac.il/volatile/coco/vgg_features/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')


cmd:option('-start_from', 0, 'iter number you want to load')
cmd:option('-co_atten_type', 'Alternating_message', 'co_attention type. Parallel or Alternating, alternating trains more faster than parallel.')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-checkpoint_dir_path', 'save/train', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-MC', 0, 'are you evaluating MultipleChoice task')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------		
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  cutorch.manualSeed(opt.seed)
--  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)


-------------------------------------------------------------------------------
-- Helper functions
-------------------------------------------------------------------------------

-- Order values (for logs)
function pairsByKeys (t, f)
      local a = {}
      for n in pairs(t) do table.insert(a, n) end
      table.sort(a, f)
      local i = 0      -- iterator variable
      local iter = function ()   -- iterator function
        i = i + 1
        if a[i] == nil then return nil
        else return a[i], t[a[i]]
        end
      end
      return iter
end


-- Copy table.
function copy(t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v:clone() end
    setmetatable(target, meta)
    return target
end

--Join CudaTensors
jtnn=nn.JoinTable(1)
if opt.gpuid >= 0 then
	jtnn = jtnn:cuda()
end
function join_vector(tensor_table)
	return jtnn:forward(tensor_table):clone();
end

--split CudaTensors
function split_vector(w,sizes)
	local tensor_table={};
	local offset=1;
	local n;
	if type(sizes)=="table" then
		n=#sizes;
	else
		n=sizes:size(1);
	end
	for i=1,n do
		table.insert(tensor_table,w[{{offset,offset+sizes[i]-1}}]);
		offset=offset+sizes[i];
	end
	return tensor_table;
end

function generate_checkpoint_path(iter)
	return path.join(opt.checkpoint_dir_path .. '_' .. opt.feature_type, 'model_id' .. opt.id .. '_iter'.. iter)
end

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
--used for logs
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if opt.start_from > 0 then
  loaded_checkpoint = torch.load(generate_checkpoint_path(opt.start_from) .. ".t7")
  lmOpt = loaded_checkpoint.lmOpt
else
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.hidden_last_size = opt.hidden_last_size
  lmOpt.rnn_hiddensize = opt.rnn_hiddensize
  lmOpt.dropout = opt.dropout / 10
  lmOpt.pw_matrix_rank = opt.pw_matrix_rank
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.output_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
  lmOpt.ques_embedding_size = opt.ques_embedding_size
  lmOpt.img_embedding_size = opt.img_embedding_size
  lmOpt.learning_rate = opt.learning_rate
end
print(lmOpt)

vqa_model = nn.vqa_model(lmOpt):cuda()
crit = nn.CrossEntropyCriterion():cuda()

--
local params, grad_params = vqa_model:getParameters()

if opt.start_from > 0 then
  print('Load the weight...')
  print(params:size(),loaded_checkpoint.params:size())   
  params:copy(loaded_checkpoint.params)  
end

print('total number of parameters in vqa_model: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage() -- just in case

-------------------------------------------------------------------------------
-- Create the Data Loader instance 
-------------------------------------------------------------------------------

local loader = DataLoader{h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type}

collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
 
vqa_model:evaluate()
  
  loader:resetIterator(split)
  local predictions = {}
  local total_num = loader:getDataNum(split)
  
  local logprob_all = torch.Tensor(total_num, lmOpt.output_size)
  local ques_id = torch.Tensor(total_num)
  
  
  for i = 1, total_num, lmOpt.batch_size do
	xlua.progress(i, total_num)
    local r = math.min(i+lmOpt.batch_size-1, total_num) 
	
    local data = loader:getBatch{batch_size = r-i+1, split = split}
	
	-- ship the data to cuda
    if opt.gpuid >= 0 then
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
      data.mc_answer = data.mc_answer:cuda()
    end
  
    local out_feat = vqa_model:forward({data.questions,data.images,data.mc_answer})
    
    if (opt.MC>0) then
        local mc_feat = torch.Tensor(out_feat:size()):fill(-99999)
        for i=1, out_feat:size(1) do
            for j=1,data.mc_answer[i]:size(1) do
                if (data.mc_answer[i][j] ~= 0) then                    
                    mc_feat[i][data.mc_answer[i][j]] = out_feat[i][data.mc_answer[i][j]] 
                end
            end
        end
        out_feat = mc_feat        
    end
	
	logprob_all:sub(i, r):copy(out_feat:float())
    ques_id:sub(i, r):copy(data.ques_id)
	
  end
  tmp,pred = torch.max(logprob_all,2)
  
  for i=1,total_num do
    local ans = loader.ix_to_ans[tostring(pred[{i,1}])]
    table.insert(predictions,{question_id=ques_id[i],answer=ans})
  end
  return {predictions}
end

predictions  = eval_split(2)
utils.write_json('vqa_OpenEnded_mscoco_test-dev2015_multmodal'..opt.id..'-'..opt.start_from..'_results'..'.json', predictions[1])


  
