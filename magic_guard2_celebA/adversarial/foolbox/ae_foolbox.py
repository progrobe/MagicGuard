import tensorflow as tf
import foolbox as fb

model = tf.keras.applications.ResNet50(weights="imagenet")
preprocessing = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])
bounds = (0, 255)
fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)

fb.utils.accuracy(fmodel, images, labels)

# attack = fb.attacks.LinfDeepFoolAttack()
# raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)