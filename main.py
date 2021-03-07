import keras

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  print("yes")
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

from keras_segmentation.models.segnet import segnet
from keras.models import load_model

model = segnet(n_classes=51 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/segnet_1" , epochs=3
)

out = model.predict_segmentation(
    inp="1_input.jpg",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)


model.save('my_model.h5')
# evaluating the model 
#from keras.models import load_model

#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')
#import keras
#import tensorflow as tf


#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
