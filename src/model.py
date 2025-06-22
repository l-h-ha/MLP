from mlp import model, layer, activations

model = model()
model.set_layers([
    layer(784, activations.relu),
    layer(128, activations.relu),
    layer(10, activations.relu)
])
model.info()