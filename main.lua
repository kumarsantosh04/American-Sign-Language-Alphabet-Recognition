require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'cudnn'
local matio = require 'matio'
----------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            '',            'Model file - must return valid network.')
cmd:option('-LR',                 0.00001,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          16,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              50,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               'vgg_19.t7',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-normalization',      'image',               'simple - whole sample, channel - by image channel, image - mean and std images')
cmd:option('-format',             'rgb',                  'rgb or yuv')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          1,                      'visualizing results')

opt = cmd:parse(arg or {})
--opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
trainingerror = torch.Tensor[50];
validationerror = torch.Tensor[50];

if opt.augment then
    require 'image'
end
----------------------------------------------------------------------
-- Model + Loss:
local model
if paths.filep(opt.load) then
  pcall(require , 'cunn')
  pcall(require , 'cudnn')
end
  --model = torch.load('googmodel.t7')
 model=torch.load('inception.t7')
 --model=nn.Sequential()
 model:remove(24)
 model:add(nn.Linear(1024,36))
 model:add(nn.LogSoftMax())
 --print(model)
 model=model:cuda()
  model:evaluate()
  collectgarbage()
--else
  --model = require(opt.network)


--print(model)
local loss = nn.ClassNLLCriterion()
-- classes
--[[local data = require 'Data'
--print(data)
local classes = data.Classes]]
print('Loading data...')
local TrainData1 = matio.load('training_data.mat')
local TestData1 = matio.load('validation_data.mat')
local TrainData = TrainData1.training_data

TrainData1 = nil
TestData = TestData1.validation_data

TestData1 = nil
l1 = TrainData.label:size()[1]
l2 = TestData.label:size()[1]
x=torch.ByteTensor(l1)
y=torch.ByteTensor(l2)
for i=1,l1 do
  x[i]=TrainData.label[i][1]
end
for j=1,l2 do
  y[j]=TestData.label[j][1]
end
TrainData.label=x
TestData.label=y
collectgarbage()
print(TrainData)
print(TestData)
print('... loading complete')
classes = {'1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12','13', '14', '15','16', '17', '18','19', '20', '21','22', '23', '24','25', '26', '27','28', '29', '30','31', '32', '33','34', '35', '36'}
----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(36,classes)

local AllowVarBatch = not opt.constBatchSize


----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local TensorType = types[opt.type] or 'torch.FloatTensor'

if opt.type == 'cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    local cudnnAvailable = pcall(require , 'cudnn')
    if cudnnAvailable then
      model = cudnn.convert(model, cudnn)
    end
elseif opt.type == 'cl' then
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.devid)
end

model:type(TensorType)
loss = loss:type(TensorType)

---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

----------------------------------------------------------------------
print '==> Network'
--print(model)
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)

------------------Optimization Configuration--------------------------
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    dampening = 0,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}
----------------------------------------------------------------------

local function SampleImages(images,labels)
    if not opt.augment then
        return images,labels
    else

        local sampled_imgs = images:clone()
        for i=1,images:size(1) do
            local sz = math.random(9) - 1
            local hflip = math.random(2)==1

            local startx = math.random(sz)
            local starty = math.random(sz)
            local img = images[i]:narrow(2,starty,32-sz):narrow(3,startx,32-sz)
            if hflip then
                img = image.hflip(img)
            end
            img = image.scale(img,32,32)
            sampled_imgs[i]:copy(img)
        end
        return sampled_imgs,labels
    end
end


------------------------------
local function Forward(Data, train)
   
  local SizeData = Data.data:size(1)

  SizeBatch = math.floor(Data.data:size(1)/opt.batchSize)

  local yt,x
  --local NumBatches = 1
  local lossVal = 0
  local num = 1;
  for NumBatches=1,SizeBatch do

    yt = Data.label[{{num,num+opt.batchSize-1}}]:cuda()

    x = Data.data[{{num,num+opt.batchSize-1},{},{},{}}]:cuda()

    local y, currLoss
   y = model:forward(x)
    currLoss = loss:forward(y,yt)
    if train then
      local function feval()
        model:zeroGradParameters()
        local dE_dy = loss:backward(y, yt)
        model:backward(x, dE_dy)
        return currLoss, Gradients
      end
      _G.optim[opt.optimization](feval, Weights, optimState)
      if opt.nGPU > 1 then
        model:syncParameters()
      end
    end

    lossVal = currLoss + lossVal

    if type(y) == 'table' then --table results - always take first prediction
      y = y[1]
    end

    confusion:batchAdd(y,yt)
    xlua.progress(NumBatches, SizeBatch)
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
    num = num + opt.batchSize
  end
  if(Data.data:size(1)%opt.batchSize ~= 0) then
    yt = Data.label[{{num,Data.data:size(1)}}]:cuda()
    x = Data.data[{{num,Data.data:size(1)},{},{},{}}]:cuda()
    y = model:forward(x)
      currLoss = loss:forward(y,yt)
      if train then
        local function feval()
          model:zeroGradParameters()
          local dE_dy = loss:backward(y, yt)
          model:backward(x, dE_dy)
          return currLoss, Gradients
        end
        _G.optim[opt.optimization](feval, Weights, optimState)
        if opt.nGPU > 1 then
          model:syncParameters()
        end
      end

      lossVal = currLoss + lossVal

      if type(y) == 'table' then --table results - always take first prediction
        y = y[1]
      end

      confusion:batchAdd(y,yt)
    end
    collectgarbage()
  return(lossVal/math.ceil(Data.data:size(1)/opt.batchSize))
end

------------------------------
local function Train(Data)  
  model:training()
  --print(Data)
  return Forward(Data, true)
end

local function Test(Data)
  model:evaluate()
  return Forward(Data, false)
end
------------------------------

local epoch = 0
print '\n==> Starting Training\n'
local validationerror=torch.Tensor(opt.epoch,1)
while epoch ~= opt.epoch do
    --data.TrainData:shuffleItems()

    print('Epoch ' .. epoch)
    --Train
    confusion:zero()
    --print(data.TrainData)
    local LossTrain = Train(TrainData)    
    torch.save(netFilename, model:clearState())
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end
    print('Training Error = ' .. ErrTrain)
    print('Training Loss = ' .. LossTrain)

    --Test
    confusion:zero()
    local LossTest = Test(TestData)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end

    print('Validation Error = ' .. ErrTest)
 --   validationerror[epoch] = ErrTest;
    print('Validation Loss = ' .. LossTest)

    Log:add{['Training Error']= ErrTrain, ['Validation Error'] = ErrTest}
  --  trainerror[epoch] = Errtrain;
    
    epoch = epoch + 1
end
if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Validation Error'] = '-'}
        Log:plot()

--torch.save('trainerror.t7',trainerror);
--torch.save('validationerror.t7',validationerror);



    end
