import torch
import torch.nn as nn
import torchvision


class ActivationExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers=None, activated_layers=None, activation_value=1):
        super().__init__()
        self.model = model
        if layers is None:
            self.layers = []
            for n, _ in model.named_modules():
                self.layers.append(n)
        else:
            self.layers = layers
        self.activations = {layer: torch.empty(0) for layer in self.layers}
        self.pre_activations = {layer: torch.empty(0) for layer in self.layers}
        self.activated_layers = activated_layers
        self.activation_value = activation_value
        
        self.hooks = []

        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))

    def get_activation_hook(self, layer_id: str):
        def fn(_, input, output):
            #self.activations[layer_id] = output.detach().clone()
            self.activations[layer_id] = output
            self.pre_activations[layer_id] = input[0]
            # modify output
            if self.activated_layers is not None and layer_id in self.activated_layers:
              for idx in self.activated_layers[layer_id]:
                for sample_idx in range(0, output.size()[0]):
                  output[tuple(torch.cat((torch.tensor([sample_idx]).to(idx.device), idx)))] = self.activation_value
            return output
        return fn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        self.model(x)
        return self.activations


class ResNet18(torchvision.models.ResNet):
    def __init__(self, num_classes, **kwargs):
        super(ResNet18, self).__init__(
            torchvision.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            num_classes,
            **kwargs
        )

    def forward(self, x):
        return super(ResNet18, self).forward(x)

    @staticmethod
    def get_relevant_layers():
        return ['bn1',
                'layer1.0.bn1', 'layer1.0.bn2', 'layer1.1.bn1', 'layer1.1.bn2',
                'layer2.0.bn1', 'layer2.0.bn2', 'layer2.1.bn1', 'layer2.1.bn2',
                'layer3.0.bn1', 'layer3.0.bn2', 'layer3.1.bn1', 'layer3.1.bn2',
                'layer4.0.bn1', 'layer4.0.bn2', 'layer4.1.bn1', 'layer4.1.bn2']


def main():
    model = ResNet18(10)

    # This way you can examine what architecture a model has,
    # and what is the name or "path" of relevant layers.
    print(next(iter(model.named_modules()))[1])

    # We create the extractor such that it observes the appropriate layers.
    extractor = ActivationExtractor(model, ResNet18.get_relevant_layers())

    # Here, you give a batch of your data to the extractor, and
    # it returns the activations in the form of a dict.
    sample_batch = torch.empty((10, 3, 32, 32)).uniform_(0.0, 1.0)  # Some input for demonstration's sake
    with torch.no_grad():
        # One way to use it:
        activations = extractor(sample_batch)
        # Another way to use it:
        model(sample_batch)
        activations = extractor.activations
    print(activations)


if __name__ == '__main__':
    main()
