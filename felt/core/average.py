"""Model for performing federated averaging of models."""
import numpy as np

ATTRIBUTE_LIST = ["coef_", "intercept_", "coefs_", "intercepts_"]


def get_models_params(models):
    """Extract trainable parameters from scikit-learn models.

    Args:
        modesl (list[object]): list of scikit-learn models.

    Returns:
        (dict[str, list[ndarray]]): dictionary mapping attributes to list of values
            numpy arrays extracted from models.
    """
    params = {}
    for param in ATTRIBUTE_LIST:
        params[param] = []
        try:
            for model in models:
                params[param].append(getattr(model, param))
        except:
            params.pop(param, None)

    return params


def set_model_params(model, params):
    """Set new values of trainable params to scikit-learn models.

    Args:
        model (object): scikit-learn model.
        params (dict[str, ndarray]): dictinary mapping attributes to numpy arrays.

    Returns:
        (object): scikit-learn model with new values.
    """
    for param, value in params.items():
        setattr(model, param, value)
    return model


def average_models(models):
    """Average trainable parameters of scikit-learn models.

    Args:
        models (list[object]): list of scikit-learn models.

    Returns:
        (object): scikit-learn model with new values.
    """
    params = get_models_params(models)
    for param, values in params.items():
        print(np.shape(values))
        #val = np.mean(values, axis= 0)
        val = np.array([np.mean(i) for i in values])
        #val, error = tolerant_mean(values)
        #mean = lambda x: sum(x)/float(len(x)) 
        #transpose = [[item[i] for item in values] for i in range(len(values[0]))]
        #val = [[mean(j[i] for j in t if i < len(j)) for i in range(len(max(t, key = len)))] for t in transpose]
        
        params[param] = val.astype(values[0].dtype)

    model = set_model_params(models[0], params)
    return model


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

