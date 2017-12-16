require 'nngraph'
require 'nn'
require 'cunn'
require 'misc.maskSoftmax'
require 'cbp.SignedSquareRoot'
require 'misc.ExpandAs'
nngraph.setDebug(true)
local attention = {}

function corrMatrix3(X_size,Y_size,Z_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 

    local X = inputs[1]
    local Y = inputs[2]
    local Z = inputs[3]

    Z_split = nn.SplitTable(1,2)(Z)
    Z_outputs = {}

    for i=1,Z_size do
        table.insert(Z_outputs, nn.View(-1,X_size,Y_size,1)(nn.MM(false, true)({X,nn.CMulTable()(nn.ExpandAs()({Y,nn.View(1,-1):setNumInputDims(1)(nn.Contiguous()(nn.SelectTable(i)(Z_split)))}))})))
    end

    output_tensor = nn.JoinTable(3,3)(Z_outputs)
    table.insert(outputs, output_tensor)
    return nn.gModule(inputs, outputs)
end


function Un(feat, seq_size, embedding_size)
	local embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(-1, embedding_size)(feat))
    local embed = nn.View(-1, seq_size, embedding_size)(embed_dim)
    
    local embed_feat = nn.Dropout(0.5)(nn.Tanh()(embed))
    local h = nn.Linear(embedding_size, 1)(nn.View(-1, embedding_size)(embed_feat))
    local P = nn.View(-1, seq_size)(h)
	
	return P
end

function Pw(feat1, feat2, seq1_size, seq2_size, embedding_size)
	-- W_Q Q
    local feat1_embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(embedding_size):setNumInputDims(2)(feat1))
    local feat1_embed = nn.View(-1, seq1_size, embedding_size):setNumInputDims(2)(feat1_embed_dim)
	
    -- W_V V
    local feat2_embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(embedding_size):setNumInputDims(2)(feat2))
    local feat2_embed = nn.View(-1, seq2_size, embedding_size):setNumInputDims(2)(feat2_embed_dim)
	
	-- QAV, this is pairwise matrix
	local weight_matrix = nn.MM(false, true)({feat1_embed,feat2_embed})
	
	local scaled_weight_matrix = nn.CMul(seq1_size,seq2_size)(weight_matrix)
	
    -- Could easily be replaced with nn.Mul, nn.Add
	local P1 = nn.Tanh()(nn.View(-1,seq1_size)(nn.Linear(seq2_size,1)(
				nn.View(-1,seq2_size)(scaled_weight_matrix))))
	local P2 = nn.Tanh()(nn.View(-1,seq2_size)(nn.Linear(seq1_size,1)(
                nn.View(-1,seq1_size)((nn.Transpose({2,3})(scaled_weight_matrix))))))
	
	return P1, P2
end


function Tr(feat1, feat2, feat3, seq1_size, seq2_size, seq3_size, embedding_size)
	-- W_Q Q
    local feat1_embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(embedding_size):setNumInputDims(2)(feat1))
    local feat1_embed = nn.View(-1, seq1_size, embedding_size):setNumInputDims(2)(feat1_embed_dim)
	
    -- W_V V
    local feat2_embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(embedding_size):setNumInputDims(2)(feat2))
    local feat2_embed = nn.View(-1, seq2_size, embedding_size):setNumInputDims(2)(feat2_embed_dim)
    
    -- W_A A
    local feat3_embed_dim = nn.Linear(embedding_size, embedding_size)(nn.View(embedding_size):setNumInputDims(2)(feat3))
    local feat3_embed = nn.View(-1, seq3_size, embedding_size):setNumInputDims(2)(feat3_embed_dim)
	
	-- this is triwise matrix
    --V x A x Q (Switch order to reduce memory consumption)
	local triwise = corrMatrix3(seq2_size, seq3_size, seq1_size)({feat2_embed,feat3_embed,feat1_embed})
	
	local scaled_weight_matrix = nn.CMul(seq1_size,seq2_size,seq3_size)(triwise)
	
    -- Could easily be replaced with nn.Mul, nn.Add
	local P1 = nn.Tanh()(nn.View(-1,seq1_size)(nn.Linear(seq2_size*seq3_size,1)(
				nn.View(-1,seq2_size*seq3_size)(nn.Transpose({1,3}):setNumInputDims(3)(scaled_weight_matrix)))))
	local P2 = nn.Tanh()(nn.View(-1,seq2_size)(nn.Linear(seq1_size*seq3_size,1)(
                nn.View(-1,seq1_size*seq3_size)((scaled_weight_matrix)))))
    local P3 = nn.Tanh()(nn.View(-1,seq3_size)(nn.Linear(seq1_size*seq2_size,1)(
                nn.View(-1,seq1_size*seq2_size)((nn.Transpose({1,2}):setNumInputDims(3)(scaled_weight_matrix))))))
	
	return P1, P2, P3
end


--Apply attention
function attend(feat, potential, feat_size, mask)
	local prob
	if(mask) then
		prob = nn.maskSoftMax()({potential,mask})
	else
		prob = nn.SoftMax()(potential)
	end
	return nn.View(-1, feat_size)(nn.MM(false,false)(
			{nn.View(1,-1):setNumInputDims(1)(prob),
			feat}))
end



function module_step(ques_feat, img_feat, mc_feat, ques_seq_size, img_seq_size, mc_seq_size, embedding_size, mask)
	local q_v_a, v_q_a, a_v_q  = Tr(ques_feat, img_feat, mc_feat, ques_seq_size, img_seq_size, mc_seq_size, embedding_size)
    local q_v, v_q  = Pw(ques_feat, img_feat, ques_seq_size, img_seq_size, embedding_size)
    local q_a, a_q  = Pw(ques_feat, mc_feat, ques_seq_size, mc_seq_size, embedding_size)
    local v_a, a_v  = Pw(img_feat, mc_feat, img_seq_size, mc_seq_size, embedding_size)
    local a = Un(mc_feat, mc_seq_size, embedding_size)
	local v = Un(img_feat, img_seq_size, embedding_size)
	local q = Un(ques_feat, ques_seq_size, embedding_size)
    
    -- Could easily be replaced with nn.Mul, nn.Add
    ques_merge_potentials = nn.View(-1, ques_seq_size)(nn.Linear(4,1)(nn.View(4):setNumInputDims(2)(nn.JoinTable(2,2)({
        nn.View(-1,1):setNumInputDims(1)(q),
        nn.View(-1,1):setNumInputDims(1)(q_v),
        nn.View(-1,1):setNumInputDims(1)(q_a),
        nn.View(-1,1):setNumInputDims(1)(q_v_a)}))))
    
    img_merge_potentials = nn.View(-1, img_seq_size)(nn.Linear(4,1)(nn.View(4):setNumInputDims(2)((nn.JoinTable(2,2)({
        nn.View(-1,1):setNumInputDims(1)(v),
        nn.View(-1,1):setNumInputDims(1)(v_q),
        nn.View(-1,1):setNumInputDims(1)(v_a),
        nn.View(-1,1):setNumInputDims(1)(v_q_a)})))))
        
    ans_merge_potentials = nn.View(-1, mc_seq_size)(nn.Linear(4,1)(nn.View(4):setNumInputDims(2)((nn.JoinTable(2,2)({
        nn.View(-1,1):setNumInputDims(1)(a),
        nn.View(-1,1):setNumInputDims(1)(a_q),
        nn.View(-1,1):setNumInputDims(1)(a_v),
        nn.View(-1,1):setNumInputDims(1)(a_v_q)})))))
        
	local ques_feat = attend(ques_feat, ques_merge_potentials, embedding_size, mask)
    local img_feat = attend(img_feat, img_merge_potentials, embedding_size)
	local ans_feat = attend(mc_feat, ans_merge_potentials, embedding_size)
	
	return ques_feat, img_feat, ans_feat
end

function attention.margin_attention(ques_seq_size, img_seq_size, mc_seq_size, embedding_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 

    

    local ques_feat = inputs[1]
    local img_feat = inputs[2]
    local mc_feat = inputs[3]
    local mask = inputs[4]


	
	local Q_atten, I_atten, A_atten = module_step(ques_feat, img_feat, mc_feat, ques_seq_size, img_seq_size, mc_seq_size, embedding_size, mask)
    
	 
    table.insert(outputs, Q_atten)
    table.insert(outputs, I_atten)
    table.insert(outputs, A_atten)
    return nn.gModule(inputs, outputs)
end

return attention

