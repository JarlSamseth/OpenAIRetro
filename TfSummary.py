import tensorflow as tf
from keras import backend as K


class TfSummary:
    def __init__(self, dir_name, summary_names) -> None:
        K.clear_session()
        self.sess = K.get_session()


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.__setup_summary(summary_names)
        self.summary_writer = tf.summary.FileWriter(
            'summary/' + dir_name, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __setup_summary(self, summary_names):

        summary_placeholders = {}
        update_ops = {}
        for name in summary_names:
            summary_var = tf.Variable(0., name=name)
            tf.summary.scalar(name, summary_var)

            summary_placeholder = tf.placeholder(tf.float32, name=name)
            summary_placeholders[name] = summary_placeholder

            update_ops[name] = summary_var.assign(summary_placeholder)

        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def add_to_summary(self, stats, episode):
        for stat_name in stats:
            self.sess.run(self.update_ops[stat_name], feed_dict={
                self.summary_placeholders[stat_name]: float(stats[stat_name])
            })
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, episode + 1)
