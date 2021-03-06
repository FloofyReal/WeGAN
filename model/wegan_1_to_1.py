from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv2d, conv3d_transpose, dis_block, linear
from utils.utils import denormalize, save_image


class WeGAN1to1(object):
    def __init__(self,
                 input_batch,
                 batch_size=64,
                 frame_size=2,
                 crop_size=32,
                 channels=1,
                 wvars='11100',
                 learning_rate=0.0002,
                 beta1=0.5,
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.channels = channels
        self.learning_rate = learning_rate
        self.frame_size = frame_size
        self.videos = input_batch
        self.wvars = wvars
        self.build_model()

    def generator(self, img_batch):
        with tf.variable_scope('g_') as vs:
            """ -----------------------------------------------------------------------------------
            ENCODER 
            ----------------------------------------------------------------------------------- """
            print('ENCODER')

            self.en_h0 = conv2d(img_batch, self.channels, 128, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())

            self.en_h1 = conv2d(self.en_h0, 128, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())

            self.en_h2 = conv2d(self.en_h1, 256, 512, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())

            self.en_h3 = conv2d(self.en_h2, 512, 1024, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            GENERATOR 
            ----------------------------------------------------------------------------------- """
            print('GENERATOR')

            self.z_ = tf.reshape(self.en_h3, [self.batch_size, 2, 2, 1024])
            print(self.z_.get_shape().as_list())

            self.fg_h1 = tf.image.resize_images(self.z_, [4,4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.fg_h1 = conv2d(self.fg_h1, 1024, 512, d_h=1, d_w=1, name="gen_conv1")
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())

            self.fg_h2 = tf.image.resize_images(self.fg_h1, [8,8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.fg_h2 = conv2d(self.fg_h2, 512, 256, d_h=1, d_w=1, name="gen_conv2")
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)
            print(self.fg_h2.get_shape().as_list())

            self.fg_h3 = tf.image.resize_images(self.fg_h2, [16,16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.fg_h3 = conv2d(self.fg_h3, 256, 128, d_h=1, d_w=1, name="gen_conv3")
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())

            self.fg_h4 = tf.image.resize_images(self.fg_h3, [self.crop_size,self.crop_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.fg_h4 = conv2d(self.fg_h4, 128, self.channels, d_h=1, d_w=1, name="gen_conv4")
            self.fg_fg = tf.nn.tanh(self.fg_h4, name='g_f_actication')
            print(self.fg_fg.get_shape().as_list())

            gen_reg = tf.reduce_mean(tf.square(img_batch - self.fg_fg))

        variables = tf.contrib.framework.get_variables(vs)
        return self.fg_fg, gen_reg, variables

    def discriminator(self, video, reuse=False):
        with tf.variable_scope('d_', reuse=reuse) as vs:
            initial_dim = self.crop_size
            video = tf.reshape(video, [self.batch_size, self.frame_size, self.crop_size, self.crop_size, self.channels])
            d_h0 = dis_block(video, self.channels, initial_dim, 'block1', reuse=reuse, ddd=True)
            d_h1 = dis_block(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse, ddd=True)
            d_h2 = dis_block(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse, ddd=True)
            d_h3 = dis_block(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse, ddd=True)
            d_h4 = dis_block(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False, ddd=True)
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, -1]), 1)
        variables = tf.contrib.framework.get_variables(vs)
        return d_h5, variables

    def build_model(self):
        print("Setting up model...")

        # input_images = First frame of video
        self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, self.channels])
        self.videos_fake, self.gen_reg, self.generator_variables = self.generator(self.input_images)

        self.fake_min = tf.reduce_min(self.videos_fake)
        self.fake_max = tf.reduce_max(self.videos_fake)

        print('Shapes of videos:')
        print('Original:')
        print(self.videos.shape)
        print('Generated:')
        print(self.videos_fake.shape)

        self.d_real, self.discriminator_variables = self.discriminator(self.videos, reuse=False)

        # merging initial frame and generated to create full forecast "video"
        self.videos_fake = tf.stack([self.input_images, self.videos_fake], axis=1)

        self.d_fake, _ = self.discriminator(self.videos_fake, reuse=True)

        self.g_cost_pure = -tf.reduce_mean(self.d_fake)

        # self.g_cost = self.g_cost_pure + 1000 * self.gen_reg

        self.d_cost = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)

        self.videos = tf.reshape(self.videos, [self.batch_size, self.frame_size, self.crop_size, self.crop_size, self.channels])
        self.videos_fake = tf.reshape(self.videos_fake, [self.batch_size, self.frame_size, self.crop_size, self.crop_size, self.channels])

        help_v = [0,0,0,0,0]
        par = 0
        for c,k in zip(self.wvars, range(5)):
            if c == '1':
                help_v[k] = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.videos[:,:,:,:,par], self.videos_fake[:,:,:,:,par]))))
                par += 1
            else:
                help_v[k] = tf.constant(0.0)

        self.rmse_temp = help_v[0]
        self.rmse_cc = help_v[1]
        self.rmse_sh = help_v[2]
        self.rmse_sp = help_v[3]
        self.rmse_geo = help_v[4]

        tf.summary.scalar('rmse_temp', self.rmse_temp)
        tf.summary.scalar('rmse_cc', self.rmse_cc)
        tf.summary.scalar('rmse_sh', self.rmse_sh)
        tf.summary.scalar('rmse_sp', self.rmse_sp)
        tf.summary.scalar('rmse_geo', self.rmse_geo)

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.videos, self.videos_fake))))

        # self.mae = tf.metrics.mean_absolute_error(self.videos_fake, self.videos)

        # error of discriminator failing to evaluate generated sample as fake - good job generator
        tf.summary.scalar("g_cost_pure", self.g_cost_pure)
        # diff between original image and created image/sequence in generator
        tf.summary.scalar("g_cost_regularizer", self.gen_reg)
        # error of - saying fake is fake and original is original (when fake == orig and orig == fake)
        tf.summary.scalar("d_cost", self.d_cost)
        
        tf.summary.scalar("RMSE_overal", self.rmse)
        # tf.summary.tensor_summary("MAE", self.mae)

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )

        dim = self.frame_size * self.crop_size * self.crop_size * self.channels

        vid = tf.reshape(self.videos, [self.batch_size, dim])
        fake = tf.reshape(self.videos_fake, [self.batch_size, dim])
        differences = fake - vid
        interpolates = vid + (alpha * differences)
        d_hat, _ = self.discriminator(tf.reshape(interpolates, [self.batch_size, self.frame_size, self.crop_size,
                                                                self.crop_size, self.channels]), reuse=True)
        gradients = tf.gradients(d_hat, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.d_penalty = 10 * gradient_penalty

        tf.summary.scalar('d_penalty', self.d_penalty)

        self.d_cost_final = self.d_cost + self.d_penalty

        tf.summary.scalar("d_cost_penalized", self.d_cost_final)

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.d_cost_final, var_list=self.discriminator_variables)
            self.g_adam_gan = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.g_cost_pure, var_list=self.generator_variables)
            self.g_adam_first = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.gen_reg, var_list=self.generator_variables)

        self.sample = self.videos_fake
        self.summary_op = tf.summary.merge_all()

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def get_feed_dict(self, session):
        images = session.run(self.videos)[:, 0, :, :, :]
        feed_dict = {self.input_images: images}
        return feed_dict

    def get_feed_dict_and_orig(self, session):
        vid = session.run(self.videos)
        images = vid[:, 0, :, :, :]
        feed_dict = {self.input_images: images}
        return feed_dict, vid

    def train(self,
              session,
              step,
              summary_writer=None,
              log_summary=False,
              sample_dir=None,
              generate_sample=False,
              meta=None):
        if log_summary:
            start_time = time.time()

        critic_itrs = self.critic_iterations

        for critic_itr in range(critic_itrs):
            session.run(self.d_adam, feed_dict=self.get_feed_dict(session))

        feed_dict = self.get_feed_dict(session)
        session.run(self.g_adam_gan, feed_dict=feed_dict)
        session.run(self.g_adam_first, feed_dict=feed_dict)

        if log_summary:
            g_loss_pure, g_reg, d_loss_val, d_pen, rmse_temp, rmse_cc, rmse_sh, rmse_sp, rmse_geo, fake_min, fake_max, summary = session.run(
                [self.g_cost_pure, self.gen_reg, self.d_cost, self.d_penalty, self.rmse_temp, self.rmse_cc, self.rmse_sh, self.rmse_sp, self.rmse_geo, self.fake_min, self.fake_max, self.summary_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: (%g + %g), discriminator_loss: (%g + %g)" % (
                time.time() - start_time, step, g_loss_pure, g_reg, d_loss_val, d_pen))
            print("RMSE - Temp: %g, CC: %g, SH: %g, SP: %g, Geo: %g" % (rmse_temp, rmse_cc, rmse_sh, rmse_sp, rmse_geo))
            print("Fake_vid min: %g, max: %g" % (fake_min, fake_max))

        if generate_sample:
            original_sequence = session.run(self.videos)
            original_sequence = original_sequence.reshape([self.batch_size, self.frame_size, self.crop_size, self.crop_size, self.channels])
            print(original_sequence.shape)
            # images = zero state of weather
            images = original_sequence[:,0,:,:,:]
            # generate forecast from state zero
            forecast = session.run(self.sample, feed_dict={self.input_images: images})

            original_sequence = denormalize(original_sequence, self.wvars, self.crop_size, self.frame_size, self.channels, meta)
            print('saving original')
            save_image(original_sequence, sample_dir, 'init_%d_image' % step)

            forecast = denormalize(forecast, self.wvars, self.crop_size, self.frame_size, self.channels, meta)
            print('saving forecast / fakes')
            save_image(forecast, sample_dir, 'gen_%d_future' % step)

    def test(self,
              session,
              step,
              summary_writer=None,
              print_rate=1,
              sample_dir=None,
              meta=None):

        feed_dict, original_sequence = self.get_feed_dict_and_orig(session)

        g_loss_pure, g_reg, d_loss_val, d_pen, rmse_temp, rmse_cc, rmse_sh, rmse_sp, rmse_geo, summary = session.run(
            [self.g_cost_pure, self.gen_reg, self.d_cost, self.d_penalty, self.rmse_temp, self.rmse_cc, self.rmse_sh, self.rmse_sp, self.rmse_geo, self.summary_op],
            feed_dict=feed_dict)

        summary_writer.add_summary(summary, step)
        original_sequence = original_sequence.reshape([1, self.frame_size, self.crop_size, self.crop_size, self.channels])
        # print(original_sequence.shape)
        # images = zero state of weather
        # generate forecast from state zero
        forecast = session.run(self.sample, feed_dict=feed_dict)
        # return all rmse-s

        denorm_original_sequence = denormalize(original_sequence, self.wvars, self.crop_size, self.frame_size, self.channels, meta)
        denorm_forecast = denormalize(forecast, self.wvars, self.crop_size, self.frame_size, self.channels, meta)

        diff = []
        for orig, gen in zip(denorm_original_sequence, denorm_forecast):
            dif = orig - gen
            diff.append(dif[:,1,:,:,:])

        if step % print_rate == 0:
            print("Step: %d, generator loss: (%g + %g), discriminator_loss: (%g + %g)" % (step, g_loss_pure, g_reg, d_loss_val, d_pen))
            print("RMSE - Temp: %g, CC: %g, SH: %g, SP: %g, Geo: %g" % (rmse_temp, rmse_cc, rmse_sh, rmse_sp, rmse_geo))

            print('saving original')
            save_image(denorm_original_sequence, sample_dir, 'init_%d_image' % step)
            print('saving forecast / fakes')
            save_image(denorm_forecast, sample_dir, 'gen_%d_future' % step)

        rmse_all = [rmse_temp, rmse_cc, rmse_sh, rmse_sp, rmse_geo]
        costs = [g_loss_pure, g_reg, d_loss_val, d_pen]

        return rmse_all, costs, diff


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradient', grad)


def add_images_summary(sequence):
    # probably needs to reorder from NHWC to NCHW
    if image.shape[-1] == 1:
        tf.summary.image('original', image[:,0,:,:,:].reshape([32,32]))
        tf.summary.image('forecast', image[:,1,:,:,:].reshape([32,32]))
    if image.shape[-1] == 5:
        tf.summary.image('original_temp', image[:,0,:,:,0].reshape([32,32]))
        tf.summary.image('original_cc', image[:,0,:,:,1].reshape([32,32]))
        tf.summary.image('original_sh', image[:,0,:,:,2].reshape([32,32]))
        tf.summary.image('original_sp', image[:,0,:,:,3].reshape([32,32]))
        tf.summary.image('original_geo', image[:,0,:,:,4].reshape([32,32]))

        tf.summary.image('forecast_temp', image[:,1,:,:,0].reshape([32,32]))
        tf.summary.image('forecast_cc', image[:,1,:,:,1].reshape([32,32]))
        tf.summary.image('forecast_sh', image[:,1,:,:,2].reshape([32,32]))
        tf.summary.image('forecast_sp', image[:,1,:,:,3].reshape([32,32]))
        tf.summary.image('forecast_geo', image[:,1,:,:,4].reshape([32,32]))

    """
    plt.imshow(img1)
    plt.savefig('init' + str(i) + '.png')
    """
