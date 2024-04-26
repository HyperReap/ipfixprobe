import torch
import torch.optim as optim

packets_count = 1
content_size = 3
flows = 16

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.flatten = torch.nn.Flatten()  # Flatten the input tensor
        self.linear = torch.nn.Linear(packets_count * content_size, 1)  # Adjusted input size to match the size of each flow's content
        # self.linear = torch.nn.Linear(30, 1)
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        y = self.flatten(x) #16*5*10
        return self.linear(y)

    @torch.jit.export
    def training_step(self, normalized_batch) -> torch.Tensor:

         # Forward pass
        output = self.forward(normalized_batch)
        output = output.squeeze()

        # Custom training step logic
        target = torch.mean(normalized_batch, dim=(1,2))  # Shape: (16, 30, 150)

        loss = torch.nn.functional.mse_loss(output, target)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return loss




model = Model()
# Export the model to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pth")

target = torch.randn(flows,packets_count,content_size)
print("target: \n" + str(target))

print("forward")
output = model.forward(target)
print(output)

print("train")

print(
scripted_model.training_step(target)
)
print("target")

tmp = torch.mean(target, dim=(1,2))  # Shape: (16, 30, 150)
tmp = tmp.unsqueeze(1)

print(tmp)
print(tmp.shape)