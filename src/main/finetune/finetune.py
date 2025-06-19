import torch

class FineTune:
    def __init__(self):
        self.train_losses, self.val_losses, track_tokens_seen = [], [], []
        self.tokens_seen, self.global_step = 0, -1

    def __calc_loss_batch(self, input, target, model, device):
        input, target = input.to(device), target.to(device)
        logits = model(input)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target)
        return loss
    
    def tuning(self, model, data, epochs, optimizer, device):

        for epoch in range(epochs):
            #set the model to train mode
            model.train()

            for input_batch, target_batch in data:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = self.__calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step()
                examples_seen += input_batch.shape[0] # New: track examples instead of tokens 
                global_step += 1 

