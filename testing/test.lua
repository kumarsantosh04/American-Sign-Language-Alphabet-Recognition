local matio=require 'matio'
require 'nn'
require 'cudnn'
model=torch.load('Net1')
data1=matio.load('testing_data.mat')
data=data1.testdata
sz = data:size(1)
batch = math.floor(sz/5)
output = torch.CudaTensor(sz,36):fill(0)
t = 1;
for i =1,batch do
	data2=data[{{t,t+4},{},{},{}}]
	model=model:cuda()
	data2=data2:cuda()
--print(data2:type())
--print(model:forward(data2):size())
	output[{{t,t+4},{}}]=model:forward(data2)
	t = t+5
end
data2=data[{{t,t+3},{},{},{}}]
model=model:cuda()
data2=data2:cuda()
output[{{t,t+3},{}}]=model:forward(data2)
output=output:float()
matio.save('output.mat',output)

