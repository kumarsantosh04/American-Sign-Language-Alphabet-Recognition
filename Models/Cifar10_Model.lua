
--require 'cunn'
require 'nn'
 model = nn.Sequential() 

-- Convolution Layers

model:add(nn.SpatialConvolution(3, 64, 5, 5 ))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(64, 128, 3, 3))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.5))

model:add(nn.SpatialConvolution(128, 256, 3, 3))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))



model:add(nn.Dropout(0.5))
model:add(nn.SpatialConvolution(256, 128, 2,2))
model:add(nn.ReLU(true))
x=torch.Tensor(2,3,224,224)
y=model:forward(x)
print(#y)
--model:add(nn.View(128))
--model:add(nn.Dropout(0.5))


--model:add(nn.Linear(128,36))
--model:add(nn.LogSoftMax())
--return model
