import tensorflow as tf
#模拟比赛时处理后数据维度：80=batch_size * doc_number, 400 = word number(for each paragraph), 64 = word_embedding_size
word_embedding = tf.random_uniform([80, 400, 64], -1,1)
mask = tf.random_uniform([80, 400], -1,1)
num_conv_layers = 2
num_blocks = 1
dropout = 0.0
units = 64
outputs = word_embedding
sublayer = 1
num_heads = 8
l = 1
total_sublayers = (num_conv_layers + 2) * num_blocks
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs
def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret
def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    l = tf.shape(x)[:-1]
    print(l)
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret,[0,2,1,3])
def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs*mask + mask_value * (1 - mask)
def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)
def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                    logits.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            logits += b
        shape = []
        if mask is not None:
            for x in logits.shape.as_list():
                if x != None:
                    shape.append(x)
                else:
                    shape.append(-1)
            shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)
def run(outputs):
    for i in range(num_blocks):
        #位置信息省略
        #卷积实现省略
        norm_fn = tf.contrib.layers.layer_norm  # tf.contrib.layers.layer_norm or noam_norm
        outputs = norm_fn(outputs, scope = "layer_norm_1", reuse = None)
        Outputs = outputs
        #mutil_head attention实现
        queries  = outputs
        memory = queries
        memory = conv(memory, 2 * units, name="memory_projection", reuse=None)
        query = conv(queries, units, name="query_projection", reuse=None)
        #得到初始化Q,K,V矩阵：维度[80,8,400,8]
        Q = split_last_dimension(query, num_heads)#[80,400,64]-->[80,400,8,8]-->[80,8,400,8],8 = heads_num
        res =  []
        for i in tf.split(memory, 2, axis=2):
            print(i)
            res.append(split_last_dimension(i, num_heads))
        K, V = res
        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head ** -0.5
        x = dot_product_attention(Q, K, V,
                                  bias=True,
                                  seq_len=None,
                                  mask=mask,
                                  is_training=True,
                                  scope="dot_product_attention",
                                  reuse=None, dropout=dropout)
        outputs = combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))
        residual = layer_dropout(outputs, Outputs, dropout * float(l) / 4)
        num_filters = 64
        reuse = None
        # Feed-forward
        outputs = tf.nn.dropout(residual, 1.0 - dropout)
        outputs = norm_fn(outputs, scope="layer_norm_2", reuse=reuse)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name="FFN_1", reuse=reuse)
        outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / 4)
