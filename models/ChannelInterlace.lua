--
--  Interlaced scanning and grouping of channels
--
--  Part of the implementation of clcNet(https://arxiv.org/abs/1712.06145), dqz, 2017
--
require 'nn'
nn_plus = nn_plus or {}
local ChannelInterlace, parent = torch.class('nn_plus.ChannelInterlace','nn.Module')

function ChannelInterlace:__init(channel, group)
  parent.__init(self)

  assert(channel >= group, "channel has to be >= group")
  assert(channel%group==0, "Channel has to be the multiples of group")

  self.channel = channel
  self.group = group
  self.f_idx = torch.LongTensor(channel)-- forward interlacing index
  self.b_idx = torch.LongTensor(channel)-- backward interlacing index

-- forward index
  local n=1
  for i=1,group do
    local p=i
    while p<=self.channel do
       self.f_idx[p] = n
       p = p+group
       n = n+1
    end
  end

-- backward index
  local group_size = channel/group
  for i=1,group do
      for j=1,group_size do
          self.b_idx[(i-1)*group_size+j] = i+(j-1)*group
      end
  end

end

-- overide updateOutput(input)
function ChannelInterlace:updateOutput(x)
  self.output:resizeAs(x)

  if x:dim()==3 then
    self.output:index(x,1,self.f_idx)
  else
    for i=1,x:size(1) do
        self.output[i]:index(x[i],1,self.f_idx)
    end
  end

  return self.output
end

-- overide updateGradInput(input, gradOutput)
function ChannelInterlace:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)

    if gradOutput:dim()==3 then
      self.gradInput:index(gradOutput,1,self.b_idx)
    else
      for i=1,gradOutput:size(1) do
          self.gradInput[i]:index(gradOutput[i],1,self.b_idx)
      end
    end

    return self.gradInput
end

function ChannelInterlace:__tostring__()
   return string.format('ChannelInterlace : #channel=%d, #group=%d, group size=%d', self.channel, self.group, self.channel/self.group)
end

return nn_plus
