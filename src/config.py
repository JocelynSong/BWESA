import configparser

import tensorflow as tf

from src.activation import Activation

__author__ = "Jocelyn"


class BweConfig(object):
    def __init__(self, filename):
        self._cf_parser = configparser.ConfigParser()
        self._cf_parser.read(filename)
        (self.activation_name, self.embedding_size, self.num_negative_samples, self.skip_window, self.num_skips,
         self.min_count, self.sub_sample, self.interactive, self.statistics_interval, self.summary_interval,
         self.checkpoint_interval, self.normalize, self.weight_embedding, self.alpha, self.beta, self.gama, self.lamb,
         self.min_len, self.max_len, self.keep_prob, self.train_gap, self.epochs_to_train, self.batch_size,
         self.concurrent_steps, self.uniform_width, self.random_seed, self.epochs_pre_train,
         self.learning_rate_decay_factor) = self.parse()
        self.activation = Activation(self.activation_name)
        self.optimizer_config = OptimizerConfig(filename)

    def parse(self):
        activation = self._cf_parser.get("function", "activation")

        embedding_size = self._cf_parser.getint("architectures", "embedding_size")
        num_negative_samples = self._cf_parser.getint("architectures", "num_negative_samples")
        skip_window = self._cf_parser.getint("architectures", "skip_window")
        num_skips = self._cf_parser.getint("architectures", "num_skips")
        min_count = self._cf_parser.getint("architectures", "min_count")
        sub_sample = self._cf_parser.getfloat("architectures", "sub_sample")
        interactive = self._cf_parser.getboolean("architectures", "interactive")
        statistics_interval = self._cf_parser.getint("architectures", "statistics_interval")
        summary_interval = self._cf_parser.getint("architectures", "summary_interval")
        checkpoint_interval = self._cf_parser.getint("architectures", "checkpoint_interval")
        normalize = self._cf_parser.getboolean("architectures", "normalize")
        weight_embedding = self._cf_parser.getfloat("architectures", "weight_embedding")
        alpha = self._cf_parser.getfloat("architectures", "alpha")
        beta = self._cf_parser.getfloat("architectures", "beta")
        gama = self._cf_parser.getfloat("architectures", "gama")
        lamb = self._cf_parser.getfloat("architectures", "lambda")
        min_len = self._cf_parser.getfloat("architectures", "min_len")
        max_len = self._cf_parser.getint("architectures", "max_len")
        keep_prob = self._cf_parser.getfloat("architectures", "keep_prob")
        train_gap = self._cf_parser.getint("architectures", "train_gap")

        epochs_to_train = self._cf_parser.getint("parameters", "epochs_to_train")
        batch_size = self._cf_parser.getint("parameters", "batch_size")
        concurrent_step = self._cf_parser.getint("parameters", "concurrent_steps")
        uniform_width = self._cf_parser.getfloat("parameters", "uniform_width")
        random_seed = self._cf_parser.getint("parameters", "random_seed")
        epochs_pre_train = self._cf_parser.getint("parameters", "epochs_pre_train")

        learning_rate_decay_factor = self._cf_parser.getfloat("optimizer", "learning_rate_decay_factor")
        return(activation, embedding_size, num_negative_samples, skip_window, num_skips, min_count, sub_sample,
               interactive, statistics_interval, summary_interval, checkpoint_interval, normalize, weight_embedding,
               alpha, beta, gama, lamb, min_len, max_len, keep_prob, train_gap, epochs_to_train, batch_size,
               concurrent_step, uniform_width, random_seed, epochs_pre_train, learning_rate_decay_factor)


class OptimizerConfig(object):
    def __init__(self, filename):
        self._cf_parser = configparser.ConfigParser()
        self._cf_parser.read(filename)
        self.name, self.param = self.parse()
        self.lr = self.param["lr"]

    def parse(self):
        name = self._cf_parser.get("optimizer", "optimizer")
        opt_param = self.get_opt_param(name)
        return name, opt_param

    def get_opt_param(self, opt_name):
        param = dict()
        if opt_name.lower() == "sgd":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif opt_name.lower() == "sgdmomentum":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
            param["momentum"] = self._cf_parser.getfloat("optimizer", "momentum")
        elif opt_name.lower() == "adagrad":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif opt_name.lower() == "adadelta":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
            param["decay_rate"] = self._cf_parser.getfloat("optimizer", "decay_rate")
        else:
            raise ValueError("No such optimizer name:%s" % opt_name)
        return param

    def get_optimizer(self, lr):
        if self.name.lower() == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif self.name.lower() == "sgdmomentum":
            return tf.train.MomentumOptimizer(lr, self.param["momentum"])
        elif self.name.lower() == "adagrad":
            return tf.train.AdagradOptimizer(lr)
        elif self.name.lower() == "adadelta":
            return tf.train.AdadeltaOptimizer(lr, self.param["decay_rate"])
        else:
            raise ValueError("No such optimizer name: %s" % self.name)




