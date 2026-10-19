import numpy as np


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        hidden = X.dot(W1) + b1
        hidden_relu = np.maximum(0, hidden)
        scores = hidden_relu.dot(W2) + b2

        if y is None:
            return scores

        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        num_train = X.shape[0]

        loss = -np.sum(np.log(probs[np.arange(num_train), y])) / num_train
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        dscores = probs.copy()
        dscores[np.arange(num_train), y] -= 1
        dscores /= num_train

        grads = {}

        grads['W2'] = hidden_relu.T.dot(dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        dhidden = dscores.dot(W2.T)
        dhidden[hidden <= 0] = 0

        grads['W1'] = X.T.dot(dhidden) + 2 * reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            indexes = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indexes]
            y_batch = y[indexes]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            if verbose and it % 100 == 0:
                print('iteration', it, 'of', num_iters, 'loss', loss)

            if it % iterations_per_epoch == 0:
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val)

                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, X):
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
