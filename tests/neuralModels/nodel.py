import torch
import torch.optim as optim

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(30, 1)
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.linear(x)

    @torch.jit.export
    def training_step(self, batch) -> torch.Tensor:
        # batch = batch.view(1, -1)

        # if batch.shape != (1, 30):
        #     raise ValueError("Input tensor must have shape (1, 30)")

        # normalize data
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)
        normalized_input = (batch - mean) / std

        #min-max normalization
        min_value = 0
        max_value = 2000

        normalized_data = (batch - min_value) / (max_value - min_value)

        # Forward pass
        output = self.forward(normalized_data)
        
        # Custom training step logic
        targ = torch.mean(normalized_data,dim=-1,keepdim=True)
        loss = torch.nn.functional.mse_loss(output, targ)
        
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return loss




model = Model()
# Export the model to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pth")

target = torch.linspace(.0, 300.0, 30)
print("target: \n" + str(target))

print("mean")

print(torch.mean(target,dim=0))

print(
scripted_model.training_step(target)
)