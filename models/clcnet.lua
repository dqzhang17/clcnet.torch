--
--  The clcNet model definition
--
--  Implementation of clcNet(https://arxiv.org/abs/1712.06145), dqz, 2017
--
--  the initialization code comes from fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--

local nn = require 'nn'
require 'cunn'
require 'models/ChannelInterlace'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)

-- clcBlock
   function clcBlock(input_n, output_n, stride, g1, g2)
      local s = nn.Sequential()
      s:add(Convolution(input_n, input_n,3,3,stride,stride,1,1, g1))
      s:add(SBatchNorm(input_n))
      s:add(nn_plus.ChannelInterlace(input_n, g1))
      s:add(Convolution(input_n, output_n,1,1,1,1,0,0, g2))
      s:add(SBatchNorm(output_n))
      s:add(ReLU(true))
      return s
   end

-- clcNet model
   model = nn.Sequential()
   -- stage 0 : 3 -> 64
      model:add(Convolution(3,32,3,3,2,2,1,1))
      model:add(SBatchNorm(32))
      model:add(ReLU(true))
      model:add(clcBlock(32,64,1, 16,2))

   -- stage 1 : 64->128
      model:add(clcBlock(64,128,2, 32,2))
      for i=1,opt.a do
          model:add(clcBlock(128,128,1, 64,2))
      end

   -- stage 2 : 128->256
      model:add(clcBlock(128,256,2, 64,2))
      for i=1,opt.b do
          model:add(clcBlock(256,256,1, 128,2))
      end

   -- stage 3 : 256->512
      model:add(clcBlock(256,512,2, 128,2))
      for i=1,opt.c do
          model:add(clcBlock(512,512,1, 256,2))
      end

   -- stage 4 : 512->2014
      model:add(clcBlock(512,1024,2, 256,2))
      for i=1,opt.d do
          model:add(clcBlock(1024,1024,1, 512,2))
      end

      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(1024):setNumInputDims(3))
      model:add(nn.Linear(1024, 1000))

-- the following is the same as ResNet initialization
-- copied from fb.resnet.torch
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
