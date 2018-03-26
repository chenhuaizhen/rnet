import tensorflow as tf
import numpy as np
import math
import json
import random

def initData(filename,w2vfilename,c2wfilename):
    with open(filename,"r") as f:
        dictionary = json.loads(f.read())
    dict = {}
    with open(w2vfilename, "r") as f:
        data = f.read()
        temp = data.split("\n")
        for i in range(len(temp[0:-2])):
            tempArray = temp[i].split()
            dict[tempArray[0]] = np.zeros(50)
            for j in range(50):
                dict[tempArray[0]][j] = float(tempArray[j + 1])
    dict2 = {}
    with open(c2wfilename, "r") as f:
        data = f.read()
        temp = data.split("\n")
        for i in range(len(temp[0:-2])):
            tempArray = temp[i].split()
            dict2[tempArray[0]] = np.zeros(50)
            for j in range(50):
                dict2[tempArray[0]][j] = float(tempArray[j + 1])
    return dictionary,dict,dict2

dictionary,dict,dict2 = initData("./train.json","./glove.6B.50d.txt","./glove.6B.50d-char.txt")

def imporveData(data):
    string = ""
    for index,i in enumerate(data):
        if(i=="'" and index+1 < len(data) and data[index+1]=="s"):
            string += " " + i
        elif(i=="," or i=="." or i=='"' or i=="?" or i=="!" or i==")" or i==";" or i == ":"
           or i=="]" or i=="}" or i=="(" or i=="[" or i=="{" or i=="<" or i==">"):
            string += " " + i + " "
        else:
            string += i
    return string

def readData():
    maxChar = 20
    w_ctx = np.zeros((batch_size,CtxWordSize,VectorSize))
    c_ctx = np.zeros((batch_size,CtxWordSize,maxChar,VectorSize))
    w_q = np.zeros((batch_size,QuestionWordSize,VectorSize))
    c_q = np.zeros((batch_size,QuestionWordSize,maxChar,VectorSize))
    a_s = np.zeros((batch_size,1))
    a_e = np.zeros((batch_size,1))
    w_q_l = np.zeros(batch_size)
    w_ctx_l = np.zeros(batch_size)
    for i in range(batch_size):
        PID = random.randint(21, 293)
        length = len(dictionary[PID]['paragraphs'])
        LID = random.randint(0,length-1)
        Qas = len(dictionary[PID]['paragraphs'][LID]['qas'])
        QID = random.randint(0,Qas-1)
        C = imporveData(dictionary[PID]['paragraphs'][LID]['context'])
        A = imporveData(dictionary[PID]['paragraphs'][LID]['qas'][QID]['answer']['text'])
        Q = imporveData(dictionary[PID]['paragraphs'][LID]['qas'][QID]['question'])

        index_ctx = -1
        for contextI in C.split():
            w_ctx_l[i] += 1.
            index_ctx += 1
            if dict.has_key(contextI.lower()):
                w_ctx[i][index_ctx] = dict[contextI.lower()]
            index_c2w = -1
            for contextII in contextI:
                index_c2w += 1
                if (index_c2w >= 20):
                    break
                if dict2.has_key(contextII.lower()):
                    c_ctx[i][index_ctx][index_c2w] = dict2[contextII.lower()]

        index_ctx = -1
        for qI in Q.split():
            w_q_l[i] += 1.
            index_ctx += 1
            if dict.has_key(qI.lower()):
                w_q[i][index_ctx] = dict[qI.lower()]
            index_c2w = -1
            for qII in qI:
                index_c2w += 1
                if (index_c2w >= 20):
                    break
                if dict2.has_key(qII.lower()):
                    c_q[i][index_ctx][index_c2w] = dict2[qII.lower()]

        answerIndex = C.find(A)
        temp = C[:answerIndex]
        start = len(temp.split())
        end = start + len(A.split()) - 1
        a_s[i] = start
        a_e[i] = end

    c_ctx = np.reshape(c_ctx, [batch_size, CtxWordSize, maxChar * VectorSize])
    c_q = np.reshape(c_q, [batch_size, QuestionWordSize, maxChar * VectorSize])
    return w_ctx,c_ctx,w_q,c_q,a_s,a_e,w_ctx_l,w_q_l

lastPID = 0
lastLID = 0
lastQID = 0
def readValidData():
    global lastPID, lastLID, lastQID,batch_size,CtxWordSize,VectorSize,maxChar,QuestionWordSize
    w_ctx = np.zeros((batch_size,CtxWordSize,VectorSize))
    c_ctx = np.zeros((batch_size,CtxWordSize,maxChar,VectorSize))
    w_q = np.zeros((batch_size,QuestionWordSize,VectorSize))
    c_q = np.zeros((batch_size,QuestionWordSize,maxChar,VectorSize))
    a_s = np.zeros((batch_size,1))
    a_e = np.zeros((batch_size,1))
    w_q_l = np.zeros(batch_size)
    w_ctx_l = np.zeros(batch_size)
    for i in range(batch_size):
        C = imporveData(dictionary[lastPID]['paragraphs'][lastLID]['context'])
        A = imporveData(dictionary[lastPID]['paragraphs'][lastLID]['qas'][lastQID]['answer']['text'])
        Q = imporveData(dictionary[lastPID]['paragraphs'][lastLID]['qas'][lastQID]['question'])

        Qas = len(dictionary[lastPID]['paragraphs'][lastLID]['qas'])
        if (lastQID+1 >= Qas):
            lastQID = 0
            lastLID += 1
        else:
            lastQID += 1
        length = len(dictionary[lastPID]['paragraphs'])
        if(lastLID >= length):
            lastLID = 0
            lastPID += 1
        if(lastPID > 20):
            lastPID = 0

        index_ctx = -1
        for contextI in C.split():
            w_ctx_l[i] += 1.
            index_ctx += 1
            if dict.has_key(contextI.lower()):
                w_ctx[i][index_ctx] = dict[contextI.lower()]
            index_c2w = -1
            for contextII in contextI:
                index_c2w += 1
                if (index_c2w >= 20):
                    break
                if dict2.has_key(contextII.lower()):
                    c_ctx[i][index_ctx][index_c2w] = dict2[contextII.lower()]

        index_ctx = -1
        for qI in Q.split():
            w_q_l[i] += 1.
            index_ctx += 1
            if dict.has_key(qI.lower()):
                w_q[i][index_ctx] = dict[qI.lower()]
            index_c2w = -1
            for qII in qI:
                index_c2w += 1
                if (index_c2w >= 20):
                    break
                if dict2.has_key(qII.lower()):
                    c_q[i][index_ctx][index_c2w] = dict2[qII.lower()]

        answerIndex = C.find(A)
        temp = C[:answerIndex]
        start = len(temp.split())
        end = start + len(A.split()) - 1
        a_s[i] = start
        a_e[i] = end

    c_ctx = np.reshape(c_ctx,[batch_size,CtxWordSize,maxChar*VectorSize])
    c_q = np.reshape(c_q,[batch_size,QuestionWordSize,maxChar*VectorSize])
    return w_ctx,c_ctx,w_q,c_q,a_s,a_e,w_ctx_l,w_q_l

def random_weight(dim_in, dim_out, name=None, stddev=1.0):
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

def random_bias(dim, name=None):
    return tf.Variable(tf.truncated_normal([dim]), name=name)

def DropoutWrappedLSTMCell(hidden_size, in_keep_prob, name=None):
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=in_keep_prob)
    return cell

def mat_weight_mul(mat, weight):
    # [batch_size, n, m] * [m, p] = [batch_size, n, p]
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    assert (mat_shape[-1] == weight_shape[0])
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

def BiRNNSequence(x,step,size):
    x = tf.reshape(x, [batch_size,step,size])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, size])
    x = tf.split(x, step)
    return x

# this function can be ignore, because this version does not use tfrecord
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "start":
                                               tf.FixedLenFeature([1], tf.float32),
                                           "end":
                                               tf.FixedLenFeature([1], tf.float32),
                                           "ctxWord":
                                               tf.FixedLenFeature([699 * 50], tf.float32),
                                           "questionWord":
                                               tf.FixedLenFeature([60 * 50], tf.float32),
                                           "ctxChar":
                                               tf.FixedLenFeature([699 * 37], tf.float32),
                                           "questionChar":
                                               tf.FixedLenFeature([60 * 37], tf.float32),
                                       })

    start = features['start']
    end = features['end']
    ctxWord = tf.reshape(features['ctxWord'],[699, 50])
    questionWord = tf.reshape(features['questionWord'],[60,50])
    ctxChar = tf.reshape(features['ctxChar'],[699, 37])
    questionChar = tf.reshape(features['questionChar'],[60, 37])

    return start, end, ctxWord, questionWord, ctxChar, questionChar

dtype = tf.float32
maxChar = 20
state_size = 45
in_keep_prob = 0.8
VectorSize = 50
CharSize = 37
EmbeddingSize = 50
QuestionWordSize = 60
CtxWordSize = 699
span_length = 20
learning_rate = 1e-3
batch_size=14
num_epochs=1
trainNum = 56204
ValidNum = 5175 // batch_size
num_batches = trainNum//batch_size
isInit = True
modelAdd = "Model/model.ckpt"

sess = tf.InteractiveSession()

ctx = tf.placeholder(dtype,[batch_size, CtxWordSize, VectorSize])
ctx_c = tf.placeholder(dtype, [batch_size, CtxWordSize, VectorSize*maxChar])
question = tf.placeholder(dtype,[batch_size, QuestionWordSize, VectorSize])
question_c = tf.placeholder(dtype,[batch_size, QuestionWordSize, VectorSize*maxChar])
answer_start = tf.placeholder(dtype, [batch_size, 1])
answer_end = tf.placeholder(dtype, [batch_size, 1])
ctx_l = tf.placeholder(dtype, batch_size)
q_l = tf.placeholder(dtype, batch_size)
keepprob = tf.placeholder_with_default(1.0, shape=(),name="dropoutprob")
ctx_l = tf.cast(ctx_l,tf.int32)
q_l = tf.cast(q_l,tf.int32)
# embedding word-char
# c_Q = []
# c_P = []
# with tf.variable_scope('char_emb_rnn') as scope:
#     char_emb_fw_cell = DropoutWrappedLSTMCell(EmbeddingSize, 1.0)
#     char_emb_bw_cell = DropoutWrappedLSTMCell(EmbeddingSize, 1.0)
#     for t in range(QuestionWordSize):
#         if t > 0:
#             tf.get_variable_scope().reuse_variables()
#         q_c_e_outputs, q_c_e_final_fw, q_c_e_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
#             char_emb_fw_cell, char_emb_bw_cell,
#             BiRNNSequence(question_c[:,t,:],VectorSize*maxChar,1),
#             dtype=dtype, scope='char_emb')
#         c_q_t = tf.concat([q_c_e_final_fw[1], q_c_e_final_bw[1]], 1) # [batch_size,2*EmbeddingSize]
#         c_Q.append(c_q_t) # QuestionWordSize * [batch_size,2*EmbeddingSize]
#     for t in range(CtxWordSize):
#         p_c_e_outputs, p_c_e_final_fw, p_c_e_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
#             char_emb_fw_cell, char_emb_bw_cell,
#             BiRNNSequence(ctx_c[:, t, :], VectorSize*maxChar, 1),
#             dtype=tf.float32, scope='char_emb')
#         c_p_t = tf.concat([p_c_e_final_fw[1], p_c_e_final_bw[1]], 1) # [batch_size,2*EmbeddingSize]
#         c_P.append(c_p_t) # CtxWordSize * [batch_size,2*EmbeddingSize]
# c_Q = tf.stack(c_Q, 1) # [batch_size,QuestionWordSize,2*EmbeddingSize]
# c_P = tf.stack(c_P, 1) # [batch_size,CtxWordSize,2*EmbeddingSize]
# print("embedding word-char is done")

C2W = random_weight(VectorSize*maxChar, VectorSize, name='C2W')
c_Q = mat_weight_mul(question_c,C2W) # [batch_size,QuestionWordSize,VectorSize]
c_P = mat_weight_mul(ctx_c,C2W) # [batch_size,CtxWordSize,VectorSize]
# connect word-char with word
eQcQ = tf.concat([question, c_Q], 2)
ePcP = tf.concat([ctx, c_P], 2)
# eQcQ = question
# ePcP = ctx
# unstacked_eQcQ = tf.unstack(eQcQ, QuestionWordSize, 1) # QuestionWordSize*[batch_size,VectorSize + 2*CharSize]
# unstacked_ePcP = tf.unstack(ePcP, CtxWordSize, 1)# CtxWordSize*[batch_size,VectorSize + 2*CharSize]
with tf.variable_scope('encoding') as scope:
    enc_fw_cell = DropoutWrappedLSTMCell(state_size, 1.0)
    enc_bw_cell = DropoutWrappedLSTMCell(state_size, 1.0)
    # q_enc_outputs, q_enc_final_fw, q_enc_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
    #     enc_fw_cell, enc_bw_cell, unstacked_eQcQ, dtype=dtype, scope='context_encoding')
    q_enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        enc_fw_cell, enc_bw_cell, eQcQ, sequence_length=q_l, dtype=dtype, scope='context_encoding')
    tf.get_variable_scope().reuse_variables()
    # p_enc_outputs, p_enc_final_fw, p_enc_final_bw = tf.contrib.rnn.static_bidirectional_rnn(
    #     enc_fw_cell, enc_bw_cell, unstacked_ePcP, dtype=dtype, scope='context_encoding')
    p_enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        enc_fw_cell, enc_bw_cell, ePcP, sequence_length=ctx_l, dtype=dtype, scope='context_encoding')
    # u_Q = tf.stack(q_enc_outputs, 1) # [batch_size,QuestionWordSize,2*state_size]
    # u_P = tf.stack(p_enc_outputs, 1) # [batch_size,CtxWordSize,2*state_size]
    q_enc_outputs = tf.concat(q_enc_outputs, 2)
    p_enc_outputs = tf.concat(p_enc_outputs, 2)
    u_Q = tf.nn.dropout(q_enc_outputs, keepprob)
    u_P = tf.nn.dropout(p_enc_outputs, keepprob)
print("connect word-char with word is done")

# Question-Passage Matching
W_uQ = random_weight(2 * state_size, state_size, name='W_uQ')
W_uP = random_weight(2 * state_size, state_size, name='W_uP')
B_v_QP = random_bias(state_size, name='B_v_QP')
B_v_SM = random_bias(state_size, name='B_v_SM')
W_vP = random_weight(state_size, state_size, name='W_vP')
W_g_QP = random_weight(4 * state_size, 4 * state_size, name='W_g_QP')

# QP_match
with tf.variable_scope('QP_match') as scope:
    QPmatch_cell = DropoutWrappedLSTMCell(state_size, keepprob)
    QPmatch_state = QPmatch_cell.zero_state(batch_size, dtype=tf.float32)

v_P = []
W_uQ_u_Q = mat_weight_mul(u_Q, W_uQ)  # [batch_size, QuestionWordSize, state_size]
for t in range(CtxWordSize):
    # Calculate c_t
    tiled_u_tP = tf.concat([tf.reshape(u_P[:, t, :], [batch_size, 1, -1])] * QuestionWordSize, 1)
    # tiled_u_tP.shape = [batch_size, QuestionWordSize, 2 * state_size]
    W_uP_u_tP = mat_weight_mul(tiled_u_tP, W_uP) # [batch_size, QuestionWordSize, state_size]

    if t == 0:
        tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP) # [batch_size, QuestionWordSize, state_size]
    else:
        tiled_v_t1P = tf.concat([tf.reshape(v_P[t - 1], [batch_size, 1, -1])] * QuestionWordSize, 1)
        # tiled_v_t1P.shape = [batch_size, QuestionWordSize, state_size]
        W_vP_v_t1P = mat_weight_mul(tiled_v_t1P, W_vP) # [batch_size, QuestionWordSize, state_size]
        tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P) # [batch_size, QuestionWordSize, state_size]
    s_t = tf.squeeze(mat_weight_mul(tanh, tf.reshape(B_v_QP, [-1, 1]))) # [batch_size, QuestionWordSize]
    a_t = tf.nn.softmax(s_t, 1) # [batch_size, QuestionWordSize]
    tiled_a_t = tf.concat([tf.reshape(a_t, [batch_size, -1, 1])] * 2 * state_size,2)
    # tiled_a_t.shape = [batch_size, QuestionWordSize, 2 * state_size]
    c_t = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1)  # [batch_size, 2 * state_size]

    # gate
    u_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(u_P[:, t, :]), c_t], 1), 1) # [batch_size, 1, 4 * state_size]
    g_t = tf.sigmoid(mat_weight_mul(u_tP_c_t, W_g_QP)) # [batch_size, 1, 4 * state_size]
    u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t)) # [batch_size, 4 * state_size]

    # QP_match
    with tf.variable_scope("QP_match"):
        if t > 0: tf.get_variable_scope().reuse_variables()
        output, QPmatch_state = QPmatch_cell(u_tP_c_t_star, QPmatch_state)
        # output.shape = [batch_size, state_size]
        v_P.append(output)

v_P = tf.stack(v_P, 1) # [batch_size, CtxWordSize, state_size]
v_P = tf.nn.dropout(v_P, keepprob)
print("Question-Passage Matching is done")

# Self-Matching Attention
W_smP1 = random_weight(state_size, state_size, name='W_smP1')
W_smP2 = random_weight(state_size, state_size, name='W_smP2')
W_g_SM = random_weight(2 * state_size, 2 * state_size, name='W_g_SM')

SM_star = []
W_p1_v_P = mat_weight_mul(v_P, W_smP1) # [batch_size, CtxWordSize, state_size]
for t in range(CtxWordSize):
    # Calculate s_t
    tiled_v_tP = tf.concat([tf.reshape(v_P[:, t, :], [batch_size, 1, -1])] * CtxWordSize, 1)
    # tiled_v_tP.shape = [batch_size, CtxWordSize, state_size]
    W_p2_v_tP = mat_weight_mul(tiled_v_tP, W_smP2) # [batch_size, CtxWordSize, state_size]

    tanh = tf.tanh(W_p1_v_P + W_p2_v_tP) # [batch_size, CtxWordSize, state_size]
    s_t = tf.squeeze(mat_weight_mul(tanh, tf.reshape(B_v_SM, [-1, 1]))) # [batch_size, CtxWordSize]
    a_t = tf.nn.softmax(s_t, 1) # [batch_size, CtxWordSize]
    tiled_a_t = tf.concat([tf.reshape(a_t, [batch_size, -1, 1])] * state_size, 2)
    # tiled_a_t.shape = [batch_size, CtxWordSize, state_size]
    c_t = tf.reduce_sum(tf.multiply(tiled_a_t, v_P), 1) # [batch_size, state_size]

    # gate
    v_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(v_P[:, t, :]), c_t], 1), 1) # [batch_size, 1, 2 * state_size]
    g_t = tf.sigmoid(mat_weight_mul(v_tP_c_t, W_g_SM)) # [batch_size, 1, 2 * state_size]
    v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t)) # [batch_size, 2 * state_size]
    SM_star.append(v_tP_c_t_star) # CtxWordSize * [batch_size, 2 * state_size]
SM_star = tf.stack(SM_star, 1) # [batch_size, CtxWordSize, 2 * state_size]

unstacked_SM_star = tf.unstack(SM_star, CtxWordSize, 1)
with tf.variable_scope('Self_match') as scope:
    SM_fw_cell = DropoutWrappedLSTMCell(state_size, 1.0)
    SM_bw_cell = DropoutWrappedLSTMCell(state_size, 1.0)
    SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell,
                                                                                   unstacked_SM_star,
                                                                                   dtype=tf.float32)
    h_P = tf.stack(SM_outputs, 1) # [batch_size, CtxWordSize, 2 * state_size]
    h_P = tf.nn.dropout(h_P, keepprob)
print("Self-Matching Attention is done")
# h_P = v_P

# Output Layer
W_ruQ = random_weight(2 * state_size, 2 * state_size, name='W_ruQ')
# W_vQ = random_weight(state_size, 2 * state_size, name='W_vQ')
W_VrQ = random_weight(QuestionWordSize, 2 * state_size,name='W_VrQ') # has same size as u_Q
B_v_rQ = random_bias(2 * state_size, name='B_v_rQ')
B_v_ap = random_bias(state_size, name='B_v_ap')
W_hP = random_weight(2 * state_size, state_size, name='W_hP')
W_ha = random_weight(2 * state_size, state_size, name='W_ha')
with tf.variable_scope('Ans_ptr') as scope:
    AnsPtr_cell = DropoutWrappedLSTMCell(2 * state_size, keepprob)
# calculate r_Q
W_ruQ_u_Q = mat_weight_mul(u_Q, W_ruQ) # [batch_size, QuestionWordSize, 2 * state_size]
W_vQ_V_rQ = W_VrQ # [QuestionWordSize, 2 * state_size]
W_vQ_V_rQ = tf.stack([W_vQ_V_rQ] * batch_size, 0) # [batch_size, QuestionWordSize, 2 * state_size]

tanh = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ) # [batch_size, QuestionWordSize, 2 * state_size]
s_t = tf.squeeze(mat_weight_mul(tanh, tf.reshape(B_v_rQ, [-1, 1]))) # [batch_size, QuestionWordSize]
a_t = tf.nn.softmax(s_t, 1) # [batch_size, QuestionWordSize]
tiled_a_t = tf.concat([tf.reshape(a_t, [batch_size, -1, 1])] * 2 * state_size, 2)
# tiled_a_t.shape = [batch_size, QuestionWordSize, 2 * state_size]
r_Q = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1) # [batch_size, 2 * state_size]
r_Q = tf.nn.dropout(r_Q, keepprob)
# r_Q as initial state of ans ptr
h_a = None
p = [None for _ in range(2)]
for t in range(2):
    W_hP_h_P = mat_weight_mul(h_P, W_hP) # [batch_size, CtxWordSize, state_size]
    if t == 0:
        h_t1a = r_Q # [batch_size, 2 * state_size]
    else:
        h_t1a = h_a
    tiled_h_t1a = tf.concat([tf.reshape(h_t1a, [batch_size, 1, -1])] * CtxWordSize, 1)
    # tiled_h_t1a.shape = [batch_size, CtxWordSize, 2 * state_size]
    W_ha_h_t1a = mat_weight_mul(tiled_h_t1a, W_ha) # [batch_size, CtxWordSize, state_size]

    tanh = tf.tanh(W_hP_h_P + W_ha_h_t1a) # [batch_size, CtxWordSize, state_size]
    s_t = tf.squeeze(mat_weight_mul(tanh, tf.reshape(B_v_ap, [-1, 1]))) # [batch_size, CtxWordSize]
    a_t = tf.nn.softmax(s_t, 1) # [batch_size, CtxWordSize]
    p[t] = a_t

    tiled_a_t = tf.concat([tf.reshape(a_t, [batch_size, -1, 1])] * 2 * state_size,2)
    # tiled_a_t.shape = [batch_size, CtxWordSize, 2 * state_size]
    c_t = tf.reduce_sum(tf.multiply(tiled_a_t, h_P), 1) # [batch_size, 2 * state_size]

    if t == 0:
        AnsPtr_state = AnsPtr_cell.zero_state(batch_size, dtype=tf.float32)
        h_a, _ = AnsPtr_cell(c_t, (AnsPtr_state, r_Q)) # h_a.shape = [2, batch_size, 2 * state_size]
        h_a = h_a[1]

p1 = p[0]
p2 = p[1]

print("output layer is done")

# calculate loss
answer_si_idx = tf.reshape(tf.cast(answer_start, tf.int32),[-1])
answer_ei_idx = tf.reshape(tf.cast(answer_end, tf.int32),[-1])
batch_idx = tf.reshape(tf.range(0, batch_size), [-1, 1])
answer_si_re = tf.reshape(answer_si_idx, [-1, 1])
batch_idx_si = tf.concat([batch_idx, answer_si_re], 1)
answer_ei_re = tf.reshape(answer_ei_idx, [-1, 1])
batch_idx_ei = tf.concat([batch_idx, answer_ei_re], 1)

log_prob = tf.multiply(tf.gather_nd(p1, batch_idx_si), tf.gather_nd(p2, batch_idx_ei))
with tf.name_scope("loss"):
    loss = -tf.reduce_sum(tf.log(log_prob + 1e-7))
    tf.summary.scalar('loss', loss)

# Search
prob = []
search_range = CtxWordSize - span_length
for i in range(search_range):
    for j in range(span_length):
        prob.append(tf.multiply(p1[:, i], p2[:, i + j]))
prob = tf.stack(prob, axis=1) # [batch_size, search_range*span_length]
argmax_idx = tf.argmax(prob, axis=1)
pred_si = argmax_idx / span_length
pred_ei = pred_si + tf.mod(argmax_idx, span_length)
correct = tf.logical_and(tf.equal(tf.cast(pred_si, tf.int64), tf.cast(answer_si_idx, tf.int64)),
                         tf.equal(tf.cast(pred_ei, tf.int64), tf.cast(answer_ei_idx, tf.int64)))
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

print('Model built')

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

print("train built")

# start, end, ctxWord, questionWord, ctxChar, questionChar = read_and_decode("../W2VFinaltrain.tfrecords")
# min_after_dequeue = 50
# capacity = min_after_dequeue + 3 * batch_size
# startBatch, endBatch, ctxWordBatch, questionWordBatch, ctxCharBatch, questionCharBatch = \
#     tf.train.shuffle_batch([start, end, ctxWord, questionWord,ctxChar, questionChar],
#     batch_size=batch_size, capacity=capacity,
#     min_after_dequeue=min_after_dequeue)
#
# print("trainData loaded")
#
# startV, endV, ctxWordV, questionWordV, ctxCharV, questionCharV = read_and_decode("../W2VFinalvalid.tfrecords")
# startBatchV, endBatchV, ctxWordBatchV, questionWordBatchV, ctxCharBatchV, questionCharBatchV = \
#     tf.train.batch([startV, endV, ctxWordV, questionWordV,ctxCharV, questionCharV],
#     batch_size=batch_size, capacity=3*batch_size)
#
# print("validData loaded")

saver = tf.train.Saver()
if(isInit):
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, modelAdd)

print("init done")

# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter("logs/",sess.graph)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# print("start train")

def getValidAccuracy():
    acc = 0
    for i in range(ValidNum):
        # w_ctx, c_ctx, w_q, c_q, a_s, a_e
        a, b, c, d, e, f, g, h = readValidData()
        train_accuracy = accuracy.eval(feed_dict={
            ctx: a,
            ctx_c: b,
            question: c,
            question_c: d,
            answer_start: e,
            answer_end: f,
            ctx_l: g,
            q_l: h,
            keepprob:1.0
        })
        acc += train_accuracy
    return acc/ValidNum

ValidMax = 0
for epoch_idx in range(num_epochs):
    print("New data, epoch", epoch_idx)
    for batch_idx in range(num_batches):
        a, b, c, d, e, f, g, h = readData()
        _total_loss, _train_step, _acc, \
        _o1, _o2, _o3, _o4,_o5 = sess.run(
            [loss, train_step,  accuracy,
             pred_si, pred_ei, answer_si_idx, answer_ei_idx,argmax_idx],
            feed_dict={
                ctx: a,
                ctx_c: b,
                question: c,
                question_c: d,
                answer_start: e,
                answer_end: f,
                ctx_l: g,
                q_l: h,
                keepprob: 0.8
            })
        print(_o1)
        print(_o3)
        print(_o2)
        print(_o4)
        print(_o5)
        if batch_idx%500 == 0:
            # writer.add_summary(_merge, batch_idx)
            validAcc = getValidAccuracy()
            if(validAcc > ValidMax):
                ValidMax = validAcc
                saver.save(sess, modelAdd)
            print("Epoch:",epoch_idx,"Step:",batch_idx, "loss:", _total_loss, "acc:", _acc)
            print("ValidAcc:",validAcc)
        else:
            print("Epoch:",epoch_idx,"Step:", batch_idx, "loss:", _total_loss, "acc:", _acc)

# coord.request_stop()
# coord.join(threads)
