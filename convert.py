import tensorflow as tf

model = tf.keras.models.load_model("model.h5", compile=False)

# ENSURE correct format for deployment — SavedModel
tf.saved_model.save(model, "saved_model")
print("SavedModel export complete ✅")
