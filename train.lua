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
cmd:option('-input_img_train_h5','/volatile/home/idansc@st.technion.ac.il/coco/vgg_features_val/vqa_data_img_vgg_MC_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','/volatile/home/idansc@st.technion.ac.il/coco/vgg_features_val/vqa_data_img_vgg_MC_test.h5','path to the h5file containing the image feature')


cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', 0, 'iter number you want to load')
cmd:option('-feature_type', 'Residual', 'VGG or Residual')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')


cmd:option('-hidden_size',512,'the hidden layer size of the model.')
cmd:option('-hidden_last_size',1024,'the last hidden layer size of the model.')
cmd:option('-hidden_combine_size',1024,'the last hidden layer size of the model.')

cmd:option('-img_embedding_size',512,'the hidden layer size of the image embedding model.')
cmd:option('-ques_embedding_size',512,'the hidden layer size of the word embedding model.')
cmd:option('-dropout',0.5,'dropout of the model')

cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-batch_size',200,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-output_size', 1000, 'number of output answers')
cmd:option('-rnn_hiddensize',{512,512},'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('-pw_matrix_rank',512,'limit pw matrix rank')


-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|adam')
cmd:option('-learning_rate',2e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 300, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-weight_decay',0,'weight decay for regularization')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1200)

-- Evaluation/Checkpointing
cmd:option('-eval', 1, 'evaluate model. 1 = eval, 0 = not eval')
cmd:option('-save_checkpoint_every', 6000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_dir_path', 'save/train', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 600, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '0', 'an id identifying this run/job.')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')


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


-------------------------------------------------------------------------------
-- Create the Data Loader instance 
-------------------------------------------------------------------------------
local loader
if(opt.eval>0) then
    loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type, eval = opt.eval}
else
    loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type, eval = opt.eval}
end

---------------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
--used for logs
stats = {}
stats.acc = 0
stats.loss = 0
print('Building the model...')
local loaded_checkpoint


local lmOpt
if opt.start_from > 0 then
  loaded_checkpoint = torch.load(generate_checkpoint_path(opt.start_from) .. ".t7")
  lmOpt = loaded_checkpoint.lmOpt
else
  -- loaded hyper-parameters
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.hidden_last_size = opt.hidden_last_size
  lmOpt.hidden_combine_size = opt.hidden_combine_size
  lmOpt.rnn_hiddensize = opt.rnn_hiddensize
  lmOpt.dropout = 0.5
  lmOpt.pw_matrix_rank = opt.pw_matrix_rank
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.output_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
  lmOpt.learning_rate = opt.learning_rate
end

print(lmOpt)
vqa_model = nn.vqa_model(lmOpt):cuda()
crit = nn.CrossEntropyCriterion():cuda()

--
local params, grad_params = vqa_model:getParameters()

if opt.start_from > 0 then
  print('Load the weight...') 
  params:copy(loaded_checkpoint.params)

end

print('total number of parameters in vqa_model: ', params:nElement())
collectgarbage() -- just in case

-------------------------------------------------------------------------------
-- Objective Function
-------------------------------------------------------------------------------
local iter = opt.start_from
local function objFun(x)
  --load x to net parameters--
  if params~=x then
  	params:copy(x) 		
  end
  --clear gradients--
  vqa_model:training()
  grad_params:zero() 

  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
  -- PASS TO CUDA
  if opt.gpuid >= 0 then
    data.answer = data.answer:cuda()
    data.mc_answer = data.mc_answer:cuda()
    data.questions = data.questions:cuda()
    data.ques_len = data.ques_len:cuda()
    data.images = data.images:cuda()
  end
  
  local out_feat = vqa_model:forward({data.questions,data.images,data.mc_answer})
  local loss = crit:forward(out_feat, data.answer)
  
  -- evaluate acc over train 
  local tmp,pred=torch.max(out_feat,2)
  --calculate batch accuracy
  local right_sum = 0
  local acc = 0
  for i = 1, pred:size()[1] do
    if pred[i][1] == data.answer[i] then
      right_sum = right_sum + 1
    end
  end
  acc = right_sum/opt.batch_size
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = crit:backward(out_feat, data.answer)
  local dummy = vqa_model:backward({data.questions,data.images, data.mc_answer}, dlogprobs)
  -----------------------------------------------------------------------------

  --global stats for logs
  stats.acc = acc
  stats.loss = loss
  --return f(x),dfdx
  return loss, grad_params
  
end




-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------

local function eval_split(split)
  vqa_model:evaluate()
  loader:resetIterator(split)

  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local n = 0
  local predictions = {}
  local total_num = loader:getDataNum(split)
  
  --Creates JSON file for VQA eval.
  local logprob_all = torch.Tensor(total_num, opt.output_size)
  local ques_id = torch.Tensor(total_num)
  for i = 1, total_num, opt.batch_size do
    local r = math.min(i+opt.batch_size-1, total_num) 
    local data = loader:getBatch{batch_size = r-i+1, split = split}
	-- ship the data to cuda
    if opt.gpuid >= 0 then
      data.answer = data.answer:cuda()
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
      data.mc_answer = data.mc_answer:cuda()
    end
	n = n + data.images:size(1)
	
	--(Optional) Print evaluation progress 
	--xlua.progress(n, total_num)
	
    local out_feat = vqa_model:forward({data.questions,data.images,data.mc_answer})

	logprob_all:sub(i, r):copy(out_feat:float())
    ques_id:sub(i, r):copy(data.ques_id)
   end
   tmp,pred=torch.max(logprob_all,2);

   for i=1,total_num do
     local ans = loader.ix_to_ans[tostring(pred[{i,1}])]
     table.insert(predictions,{question_id=ques_id[i],answer=ans})
   end
   
   utils.write_json('OpenEnded_mscoco_high-order_results_'..opt.id..'.json', predictions)
   
   --Caluculate val acc and loss
   --Duplicate code, cuz I'm lazy
   local n = 0	
   -- naive accuracy evaluation (strict)
   while true do
       local data = loader:getBatch{batch_size = opt.batch_size, split = split}
       
       -- ship the data to cuda
       if opt.gpuid >= 0 then
         data.answer = data.answer:cuda()
         data.mc_answer = data.mc_answer:cuda()
          data.images = data.images:cuda()
          data.questions = data.questions:cuda()
          data.ques_len = data.ques_len:cuda()
        end
        n = n + data.images:size(1)
        
        --(Optional) Print evaluation progress 
        --xlua.progress(n, total_num)
      
        local out_feat = vqa_model:forward({data.questions,data.images,data.mc_answer})

        -- forward the language model criterion
        local loss = crit:forward(out_feat, data.answer)

        local tmp,pred=torch.max(out_feat,2)

        for i = 1, pred:size()[1] do
          if pred[i][1] == data.answer[i] then
            right_sum = right_sum + 1
          end
        end

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1
        
        if n >= total_num then break end
    end

  return loss_sum/loss_evals, right_sum / total_num
end



-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------


local avg_loss_history = {}
local val_acc_history = {}
local train_acc_history = {}
local train_accuracy_history = {}
local learning_rate_history = {}
local avg_loss = 0
local avg_acc = 0
local learning_rate = lmOpt.learning_rate


--state for optimization
local optim_state = {}
if opt.start_from > 0 then
   optim_state = copy(loaded_checkpoint.optim_state)
end
while true do
  --Average loss and acc (on train)
  avg_loss = avg_loss + stats.loss
  avg_acc = avg_acc + stats.acc

  if iter % opt.losses_log_every == 0 then
    avg_loss = avg_loss / opt.losses_log_every
	avg_acc = avg_acc / opt.losses_log_every
    avg_loss_history[iter] = avg_loss
    train_acc_history[iter] = avg_acc
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, avg_loss, avg_acc, learning_rate, timer:time().real))
    avg_loss = 0
	avg_acc = 0
  end

  -------------------------------------------------------------------------------
  -- Optimization
  -------------------------------------------------------------------------------
  local config = {}
  config.learningRate = learning_rate
  config.epsilon = opt.optim_epsilon
  config.weightDecay = opt.weight_decay --for regularization.
  
  if opt.optim == 'rmsprop' then
	config.alpha = opt.optim_alpha  
    local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
    -- decay the learning rate
    if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
      learning_rate = learning_rate * decay_factor -- set the decayed rate
	  lmOpt.learning_rate = learning_rate
	  config.learningRate = learning_rate
    end
    optim.rmsprop(objFun, params, config, optim_state)
  elseif opt.optim == 'adam' then
	config.beta1 = opt.optim_alpha --optim_alpha playes diffrent role here
	config.beta2 = opt.optim_beta 
    optim.adam(objFun, params, config, optim_state)
  else	
    error('bad option opt.optim')
  end
  
  -------------------------------------------------------------------------------
  -- Checkpointing
  -------------------------------------------------------------------------------
  	if (iter % opt.save_checkpoint_every == 0 and iter~=0 or iter == opt.max_iters) then	
      torch.save(generate_checkpoint_path(iter) ..'.t7', {params=params, lmOpt=lmOpt, optim_state=optim_state}) 
	end
    
    
  -------------------------------------------------------------------------------
  -- Evaluation
  -------------------------------------------------------------------------------
  if (opt.eval > 0 and (iter % opt.save_checkpoint_every == 0 and iter~=0 or iter == opt.max_iters)) then	    
      local val_loss, val_accu = eval_split(2)
	  print(os.execute("python vqa.py"))
      val_acc_history[iter] = tonumber(val_accu)
	  print('validation loss: ', val_loss, 'accuracy ', val_accu)
	  
      local checkpoint_dir_path = path.join(opt.checkpoint_dir_path .. '_' .. opt.atten_type, 'checkpoint' .. '.json')
      local checkpoint = {}      
      
      
      if(opt.start_from == iter) then
          checkpoint = utils.read_json(checkpoint_dir_path)
      else 
          checkpoint.opt = opt
          checkpoint.iter = iter
          checkpoint.loss_history = loss_history
          checkpoint.learning_rate_history = learning_rate_history
      end
	  -- Save plots
	  local valset={}
	  local n=0
	  local msg_string = 'This is an automatic message\n Accuracy history:\n'
	  local plt_path_train = path.join(opt.checkpoint_dir_path .. '_' .. opt.atten_type, 'train_acc_plot'  .. opt.id .. '.pdf')
	  local plt_path_loss = path.join(opt.checkpoint_dir_path .. '_' .. opt.atten_type , 'loss_plot' .. opt.id .. '.pdf')
	  local plt_path_val = path.join(opt.checkpoint_dir_path .. '_' .. opt.atten_type, 'val_acc_plot' .. opt.id .. '.pdf')	
	  
      msg_string = msg_string .. '\ntrain accuracy history:'
	  for k,v in pairsByKeys(train_acc_history) do
	   n=n+1
	   valset[n]=v
	   msg_string = msg_string .. '\n '.. tostring(v)
	  end
	  
	  gnuplot.pdffigure(plt_path_train)
	  gnuplot.plot({'Accuracy on train, step='..opt.save_checkpoint_every,torch.Tensor(valset)})
	  gnuplot.xlabel('iter')
	  gnuplot.ylabel('Accuracy')
	  gnuplot.movelegend('right','bottom')
	  gnuplot.plotflush()
	  
	  local valset={}
	  local n=0
	  msg_string =  msg_string .. '\navg_loss_history:'
	  for k,v in pairsByKeys(avg_loss_history) do
	   n=n+1
	   valset[n]=v
	   msg_string = msg_string .. '\n '  .. tostring(v)
	  end
	  

	  gnuplot.plot({'Loss',torch.Tensor(valset)})
	  gnuplot.xlabel('iter')
	  gnuplot.ylabel('Loss')
	  gnuplot.plotflush()
	  local valset={}
	  local n=0
	  msg_string = msg_string .. '\nValidation Accuracy:'
	  for k,v in pairsByKeys(val_acc_history) do
	   n=n+1
	   valset[n]=v
	   msg_string = msg_string .. '\n' .. tostring(v)
	  end
      
 
	  print(valset)
	
	  gnuplot.pdffigure(plt_path_val)
	  gnuplot.plot({'Current Model', torch.Tensor(valset)})
	  gnuplot.xlabel('iter')
	  gnuplot.ylabel('Val Accuracy')
	  gnuplot.movelegend('right','bottom')
	  gnuplot.plotflush()
  end
  
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
  iter = iter + 1
end
