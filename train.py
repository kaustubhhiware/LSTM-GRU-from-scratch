import argparse
import numpy as np
import tensorflow as tf
import data_loader
import os
import requests
from module import LSTM_cell, GRU_cell
# from ppretty import ppretty

'''
Name: Kaustubh Hiware
@kaustubhhiware
run with python2

python train.py --train: Defaults to LSTM, hidden_unit 32, 30 iterations / epochs
python train.py --train --hidden_unit 32 --model lstm --iter 5: Train LSTM and dump weights
python train.py --test --hidden_unit 32 --model lstm: Test with stored weights
'''

# Network Parameters
seed = 123
input_nodes = 28
output_nodes = 10
learning_rate = 0.005
num_iterations = 30
batch_size = 100
data = data_loader.DataLoader()
weights_folder = '/weights/'

# check if needed files are present or not. Downloads if needed.
def check_download_weights(model, hidden_unit):

    url_prefix = 'https://raw.githubusercontent.com/kaustubhhiware/LSTM-GRU-from-scratch/master'
    files = ['checkpoint', 'model', 'model.ckpt.data-00000-of-00001','model.ckpt.index', 'model.ckpt.meta']

    for file in files:
        if not os.path.exists(os.getcwd() + weights_folder + file):
            print 'Downloading', file
            url = url_prefix + weights_folder + file
            # urllib.urlretrieve(url, filename= os.getcwd() + weights_folder + file)
            r = requests.get(url)
            open(os.getcwd() + weights_folder + file, 'wb').write(r.content)


def SGD(train, test, hidden_unit, model, alpha=learning_rate, isTrain=False, num_iterations=num_iterations, batch_size=100):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    (trainX, trainY) = train
    (testX, testY) = test
    (n_x, m, m2) = trainX.T.shape

    Y = tf.placeholder(tf.float32, shape=[None, output_nodes], name='inputs')

    if model == 'lstm':
        rnn = LSTM_cell(input_nodes, hidden_unit, output_nodes)
    else:
        rnn = GRU_cell(input_nodes, hidden_unit, output_nodes)

    outputs = rnn.get_outputs()
    prediction = tf.nn.softmax(outputs[-1])

    cost = -tf.reduce_sum(Y * tf.log(prediction))
    saver = tf.train.Saver(max_to_keep=10)

    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if not os.path.isdir(os.getcwd() + weights_folder):
        print 'Missing folder made'
        os.makedirs(os.getcwd() + weights_folder)

    if isTrain:
        num_minibatches = len(trainX) / batch_size
        for iteration in range(num_iterations):
            iter_cost = 0.
            batch_x, batch_y = data.create_batches(trainX, trainY, batch_size=batch_size)

            for (minibatch_X, minibatch_Y) in zip(batch_x, batch_y):
                _, minibatch_cost, acc = sess.run([optimizer, cost, accuracy], feed_dict={rnn._inputs: minibatch_X, Y: minibatch_Y})
                iter_cost += minibatch_cost*1.0 / num_minibatches

            print "Iteration {iter_num}, Cost: {cost}, Accuracy: {accuracy}".format(iter_num=iteration, cost=iter_cost, accuracy=acc)

        # print ppretty(rnn)
        Train_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: trainX, Y: trainY}))
        # Test_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: testX, Y: testY}))

        save_path = saver.save(sess, "." + weights_folder + "model_" + model + "_" + str(hidden_unit) + ".ckpt")
        print "Parameters have been trained and saved!"
        print("\rTrain Accuracy: %s" % (Train_accuracy))

    else:  # test mode
        # no need to download weights in this assignment
        # check_download_weights(model, hidden_unit)

        saver.restore(sess, "." + weights_folder + "model_" + model + "_" + str(hidden_unit) + ".ckpt")
        acc = sess.run(accuracy, feed_dict={rnn._inputs: testX, Y: testY})
        print "Test Accuracy:"+"{:.3f}".format(acc)

    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help="Initiate training phase and store weights")
    parser.add_argument('--test', action="store_true", help="Initiate testing phase, load model and print accuracy")
    parser.add_argument('--hidden_unit', action="store", dest="hidden_unit", type=int, choices=[32, 64, 128, 256], help="Specify hidden unit size")
    parser.add_argument('--model', action="store", dest="model", choices=["lstm", "gru"], help="Specify model name")
    parser.add_argument('--iter', action="store", dest="iter", type=int, help="Specify number of iterations")

    trainX, trainY = data.load_data('train')
    train = (trainX, trainY)
    testX, testY = data.load_data('test')
    test = (testX, testY)
    isTrain_ = False
    num_iterations_ = num_iterations
    hidden_unit = 32
    model = 'lstm'  # lstm or gru
    args = parser.parse_args()
    if args.hidden_unit:
        print "> hidden unit flag has set value", args.hidden_unit
        hidden_unit = args.hidden_unit

    if args.model:
        print "> model flag has set value", args.model
        model = args.model

    if args.train:
        print "> Now Training"
        isTrain_ = True
        if args.iter:
            num_iterations_ = args.iter

    elif args.test:
        print "> Now Testing"
    else:
        print "> Need to provide train / test flag!"
        exit(0)

    print "> Running for", num_iterations_,"iterations"
    print "> Hidden size unit", hidden_unit
    SGD(train, test, isTrain=isTrain_, num_iterations=num_iterations_, hidden_unit=hidden_unit, model=model)


if __name__ == '__main__':
    main()
