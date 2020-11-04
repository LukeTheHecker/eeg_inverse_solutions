import tensorflow as tf


def load_convdip(pth):
    json_file = open(pth + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(pth + "/model.h5")
    print("Loaded model from disk")
    return model