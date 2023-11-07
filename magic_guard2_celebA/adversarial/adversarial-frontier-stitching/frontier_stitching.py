import tensorflow as tf
from helpers import binomial


def fast_gradient_signed(x, y, model, eps):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        loss = model.loss(y, y_pred)
    gradient = tape.gradient(loss, x)
    sign = tf.sign(gradient)
    return x + eps * sign


def gen_adversaries(model, l, dataset, eps):
    true_advs = []
    false_advs = []
    max_true_advs = max_false_advs = l // 2
    for x, y in dataset:
        # generate adversaries
        x_advs = fast_gradient_signed(x, y, model, eps) # x => (32, 224, 224, 3)
        y_preds = tf.argmax(model(x), axis=1)
        y_pred_advs = tf.argmax(model(x_advs), axis=1)
        for x_adv, y_pred_adv, y_pred, y_true in zip(x_advs, y_pred_advs, y_preds, y):
            y_true_lbl = tf.argmax(y_true)
            # x_adv is a true adversary（预测结果出错）
            # true adversary 原本没加扰动的pred是正确的，加完扰动的pred不正确
            # 把label换成原本正确的类放到模型中训练
            if y_pred == y_true_lbl and y_pred_adv != y_true_lbl and len(true_advs) < max_true_advs:
                true_advs.append((x_adv, y_true))

            # x_adv is a false adversary
            if y_pred == y_true_lbl and y_pred_adv == y_true_lbl and len(false_advs) < max_false_advs:
                false_advs.append((x_adv, y_true))

            if len(true_advs) == max_true_advs and len(false_advs) == max_false_advs:
                return true_advs, false_advs

    return true_advs, false_advs


# finds a value for theta (maximum number of errors tolerated for verification)
def find_tolerance(key_length, threshold):
    theta = 0
    factor = 2 ** (-key_length)
    s = 0
    while(True):
        # for z in range(theta + 1):
        s += binomial(key_length, theta)
        if factor * s >= threshold:
            return theta
        theta += 1


def verify(model, key_set, threshold=0.05):
    m_k = 0
    length = 0
    for x, y in key_set:
        length += len(x)
        preds = tf.argmax(model(x), axis=1)
        y = tf.argmax(y, axis=1)
        m_k += tf.reduce_sum(tf.cast(preds != y, tf.int32))
    theta = find_tolerance(length, threshold)
    m_k = m_k.numpy()
    return {
        "success": m_k < theta,
        "false_preds": m_k,
        "max_fals_pred_tolerance": theta
    }

def test_acc(model, key_set):
    count = 0
    true = 0
    for x, y in key_set:
        count = count+1
        preds = tf.argmax(model(x), axis=1)
        y = tf.argmax(y, axis=1)
        if preds == y:
            true = true+1
    return true/count