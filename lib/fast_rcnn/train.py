# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from caffe.proto import caffe_pb2
import google.protobuf.text_format
import google.protobuf as pb2


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
	    #if cfg.TRAIN.STEP:
		#    target_net=caffe.Net('models/pascal_voc/VGG16/faster_rcnn_end2end/TDM_64_concat/test.prototxt',\
        #                        'output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_TDM_64_concat_pre_l3_iter_120000.caffemodel',\
        #                         caffe.TEST)

        #if cfg.TRAIN.RESNET:
        #        vgg_net=caffe.Net('models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt',\
        #                        'output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_final_model.caffemodel',\
        #                        caffe.TEST)

        self.solver_param = caffe_pb2.SolverParameter()

	    #if cfg.TRAIN.STEP:
      	#	self.solver.net.params['L3'][0].data[:]=np.copy(target_net.params['L3'][0].data[:])
      	#	self.solver.net.params['L3'][1].data[:]=np.copy(target_net.params['L3'][1].data[:])
      	#	self.solver.net.params['dconv4_3'][0].data[:]=np.copy(target_net.params['dconv4_3'][0].data[:])
      	#	self.solver.net.params['dconv4_3'][1].data[:]=np.copy(target_net.params['dconv4_3'][1].data[:])
        #	self.solver.net.params['convT4_3'][0].data[:]=np.copy(target_net.params['convT4_3'][0].data[:])
      	#	self.solver.net.params['convT4_3'][1].data[:]=np.copy(target_net.params['convT4_3'][1].data[:])

        #if cfg.TRAIN.RESNET:
        #        self.solver.net.params['rpn_conv/3x3'][0].data[:]=np.copy(vgg_net.params['rpn_conv/3x3'][0].data[:])
        #        self.solver.net.params['rpn_conv/3x3'][1].data[:]=np.copy(vgg_net.params['rpn_conv/3x3'][1].data[:])
        #        self.solver.net.params['rpn_cls_score'][0].data[:]=np.copy(vgg_net.params['rpn_cls_score'][0].data[:])
        #        self.solver.net.params['rpn_cls_score'][1].data[:]=np.copy(vgg_net.params['rpn_cls_score'][1].data[:])
        #        self.solver.net.params['rpn_bbox_pred'][0].data[:]=np.copy(vgg_net.params['rpn_bbox_pred'][0].data[:])
        #        self.solver.net.params['rpn_bbox_pred'][1].data[:]=np.copy(vgg_net.params['rpn_bbox_pred'][1].data[:])
        #        self.solver.net.params['bbox_pred'][0].data[:]=np.copy(vgg_net.params['bbox_pred'][0].data[:])
        #        self.solver.net.params['bbox_pred'][1].data[:]=np.copy(vgg_net.params['bbox_pred'][1].data[:])
        #        self.solver.net.params['cls_score'][0].data[:]=np.copy(vgg_net.params['cls_score'][0].data[:])
        #        self.solver.net.params['cls_score'][1].data[:]=np.copy(vgg_net.params['cls_score'][1].data[:])

        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        net = self.solver.net


        def gen_data(t=0):
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            frcn_loss_cls = 0
            frcn_loss_bbox = 0
            accuarcy=0
            while self.solver.iter < max_iters:
                # Make one SGD update
                t = self.solver.iter

                timer.tic()
                self.solver.step(1)
                timer.toc()
                rpn_loss_cls += net.blobs['rpn_cls_loss'].data
                rpn_loss_bbox += net.blobs['rpn_loss_bbox'].data
                frcn_loss_cls += net.blobs['loss_cls'].data
                frcn_loss_bbox += net.blobs['loss_bbox'].data
                accuarcy+=net.blobs['accuarcy'].data
                if self.solver.iter % (10 * self.solver_param.display) == 0:
                    print 'speed: {:.3f}s / iter'.format(timer.average_time)

                if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = self.solver.iter
                    model_paths.append(self.snapshot())
                if self.solver.iter % cfg.TRAIN.DRAW_ITERS == 0:
                    yield t, rpn_loss_cls  / cfg.TRAIN.DRAW_ITERS ,rpn_loss_bbox  / cfg.TRAIN.DRAW_ITERS, frcn_loss_cls  / cfg.TRAIN.DRAW_ITERS ,frcn_loss_bbox  / cfg.TRAIN.DRAW_ITERS,accuarcy / cfg.TRAIN.DRAW_ITERS
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0
                    frcn_loss_cls = 0
                    frcn_loss_bbox = 0
                    accuarcy=0
		if self.solver.iter==max_iters:
	            time.sleep(5)
		    plt.close(fig)

        def init1():
            ax1.set_ylim(0,1)
            ax1.set_xlim(0,100)
            ax2.set_ylim(0,1)
            ax2.set_xlim(0,100)
            ax3.set_ylim(0,1)
            ax3.set_xlim(0,100)
            ax4.set_ylim(0,1)
            ax4.set_xlim(0,100)
            ax5.set_ylim(0,1)
            ax5.set_xlim(0,100)
            del xdata[:]
            del ydata1[:]
            del ydata2[:]
            del ydata3[:]
            del ydata4[:]
            del ydata5[:]
            line.set_data(xdata,ydata1)
            line2.set_data(xdata,ydata2)
            line3.set_data(xdata,ydata3)
            line4.set_data(xdata,ydata4)
            line5.set_data(xdata,ydata5)
            return line,line2,line3,line4,line5
        fig = plt.figure()
        ax1 = fig.add_subplot(5,1,1)
        ax1.set_title("RPN cls loss")
        ax2 = fig.add_subplot(5,1,2)
        ax2.set_title("RPN bbox loss")
        ax3 = fig.add_subplot(5,1,3)
        ax3.set_title("FRCN cls loss")
        ax4 = fig.add_subplot(5,1,4)
        ax4.set_title("FRCN bbox loss")
        ax5 = fig.add_subplot(5,1,5)
        ax5.set_title("ACCUARCY")
        line, = ax1.plot([], [], lw=1)
        line2, = ax2.plot([], [], lw=1)
        line3, = ax3.plot([], [], lw=1)
        line4, = ax4.plot([], [], lw=1)
        line5, = ax5.plot([], [], lw=1)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        xdata, ydata1,ydata2,ydata3,ydata4,ydata5 =[], [], [], [], [], []
        def run1(data):
            t,y1,y2,y3,y4,y5 = data
            xdata.append(t)
            ydata1.append(y1)
            ydata2.append(y2)
            ydata3.append(y3)
            ydata4.append(y4)
            ydata5.append(y5)
            xmin, xmax = ax1.get_xlim()
            if t >= xmax:
                ax1.set_xlim(xmin,2*xmax)
                ax1.figure.canvas.draw()
                ax2.set_xlim(xmin,2*xmax)
                ax2.figure.canvas.draw()
                ax3.set_xlim(xmin,2*xmax)
                ax3.figure.canvas.draw()
                ax4.set_xlim(xmin,2*xmax)
                ax4.figure.canvas.draw()
                ax5.set_xlim(xmin,2*xmax)
                ax5.figure.canvas.draw()

            line.set_data(xdata,ydata1)
            line2.set_data(xdata,ydata2)
            line3.set_data(xdata,ydata3)
            line4.set_data(xdata,ydata4)
            line5.set_data(xdata,ydata5)
            return line, line2, line3, line4,line5
        ani = animation.FuncAnimation(fig, run1, gen_data, blit=False, interval=10,
                                     repeat=False, init_func=init1)
        plt.show()
        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
