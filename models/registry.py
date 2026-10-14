REGISTRY = {}


def register(name: str, input_type: str):
    """Make a model available to the notebooks under `name`.

    input_type: "flat" (batch, n_objects * n_features) or "objects" (batch, n_objects, n_features)

    The decorated function takes ``input_shape`` (one event, without the batch dimension) and
    returns a PyTorch autoencoder (an ``nn.Module``). Registering a name again replaces the old
    model, so you can re-run a notebook cell that defines a model.
    """
    def decorator(build):
        REGISTRY[name] = {
            "build": build,
            "input_type": input_type,
        }
        return build
    return decorator
