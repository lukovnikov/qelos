import qelos as q
import torch
import numpy as np
import tensorflow as tf
from threading import current_thread
import traceback


class TFfunction(torch.autograd.Function):

    def __init__(self, forward_graph, backward_graph, session,
                 *x, **kw):
        super(TFfunction, self).__init__(*x, **kw)
        self.__inputs, self.__outputs = forward_graph
        self.__output_grads, self.__input_grads = backward_graph
        self.tfsess = session      # TODO: closing session

    def forward(self, *x):
        print("function forward current thread: {}".format(current_thread()))
        print(traceback.print_stack())
        self.save_for_backward(*x)
        _tf_out = self.tfsess.run(self.__outputs,
                                  feed_dict={
                self.__inputs: [xe.cpu().numpy() for xe in x],
                                            })
        if q.issequence(_tf_out):
            return tuple([torch.from_numpy(_tf_out_e) for _tf_out_e in _tf_out])
        else:
            return torch.from_numpy(_tf_out)

    def backward(self, *y_grad):
        print("function backward current thread: {}".format(current_thread()))
        print(traceback.print_stack())
        saved_inputs = self.saved_tensors
        # saved_inputs = map(lambda x: x.numpy(), saved_inputs)
        saved_inputs = [saved_input.numpy() for saved_input in saved_inputs]
        if q.issequence(y_grad):
            y_grad_c = y_grad[0]
            y_grad = tuple(map(lambda y_grad_e: y_grad_e.clone().cpu().numpy(), y_grad))
        else:
            y_grad_c = y_grad
            y_grad = y_grad.clone().cpu().numpy()
        # print(type(y_grad))
        all_grads = self.tfsess.run(self.__input_grads,
                                 feed_dict={self.__inputs: saved_inputs,
                                            self.__output_grads: y_grad})

        all_grads = tuple([torch_from_numpy(x_grad_e, cuda=y_grad_c) for x_grad_e in all_grads])
        x_grad = all_grads
        return x_grad


class TFModule(torch.nn.Module):
    _default_session = None

    @classmethod
    def set_default_session(cls, sess):
        cls._default_session = sess

    def __init__(self, session=None, **kw):
        super(TFModule, self).__init__(**kw)
        self._dummyparam = torch.nn.Parameter(torch.zeros(1))
        self._forward_graph = None
        self._backward_graph = None
        self._params = torch.nn.ParameterList()
        self._tfparam2param = {}
        self._param2tfparam = {}
        if session is None:
            session = self._default_session
        if session is not None:
            self._tfsess = session
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._tfsess = tf.Session(config=config)

    def flush_params_to_tensorflow(self):
        for param, tfparam in self._param2tfparam.items():
            self._tfsess.run(tfparam.assign(param.data.numpy()))

    def _build_forward_graph(self, *x):
        tfinps = create_placeholders_from_variables(*x)
        tfouts = self.apply(*tfinps)
        self._tfsess.run(tf.global_variables_initializer())
        tfparams = tf.trainable_variables()
        _tfparams = []
        for tfparam in tfparams:
            paramval = self._tfsess.run(tfparam)
            param = q.var(paramval).cuda(self._dummyparam).v
            param = torch.nn.Parameter(param.data)
            self._params.append(param)
            self._tfparam2param[tfparam.name] = param
            self._param2tfparam[param] = tfparam
            _tfparams.append(tfparam)
        self._forward_graph = (tfinps + tuple(_tfparams), tfouts)

    def _build_backward_graph(self):
        if self._forward_graph is None:
            self._build_forward_graph()
        forward_inps, forward_outs = self._forward_graph
        # make a grad placeholder for each forward out
        if q.issequence(forward_outs):
            forward_outs = forward_outs
            forward_outs_grads = tuple([tf.placeholder(forward_out.dtype, shape=forward_out.shape) for forward_out in forward_outs])
        else:
            forward_outs_grads = tf.placeholder(forward_outs.dtype, shape=forward_outs.shape)
        # make grads
        backward_outs = tf.gradients(forward_outs, forward_inps, grad_ys=forward_outs_grads)
        self._backward_graph = (forward_outs_grads, backward_outs)

    def forward(self, *x):
        if self._forward_graph is None:
            self._build_forward_graph(*x)
        if self._backward_graph is None:
            self._build_backward_graph()
        fun = TFfunction(self._forward_graph,
                         self._backward_graph,
                         self._tfsess)
        print("module forward current thread: {}".format(current_thread()))
        params = self._params
        params = [param for param in params]
        out = fun(*(x + tuple(params)))
        return out

    def apply(self, *x):         # x is a tensorflow placeholder
        raise NotImplemented()


def torch_from_numpy(x, cuda=False):     # TODO: segmentation fault when torch.from_numpy(...) in backward() of autograd.Function
    ret = q.var(torch.zeros(x.shape)).cuda(cuda).v
    # print(ret)
    # print(x)
    retdata = ret.data.numpy()
    retdata += x
    return ret.data


def create_placeholders_from_variables(*x):
    typemap = {torch.FloatTensor: tf.float32}
    tfxs = [tf.placeholder(typemap[type(xe.data)], shape=xe.size()) for xe in x]
    return tuple(tfxs)



class DummyTFModule(TFModule):
    def apply(self, x):
        return x + 1


class DummyTFModule2(TFModule):
    def apply(self, x, a):
        """ in tensorflow domain """
        y = x + a
        param = tf.get_variable("tf_param", [4, 3])
        z = y * param
        return z, y


def test():
    from scipy.signal import convolve2d, correlate2d
    from torch.nn.modules.module import Module
    from torch.nn.parameter import Parameter

    class ScipyConv2dFunction(torch.autograd.Function):
        def forward(self, input, filter):
            result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
            self.save_for_backward(input, filter)
            return torch.FloatTensor(result)

        def backward(self, grad_output):
            input, filter = self.saved_tensors
            grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
            grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')
            return torch.FloatTensor(grad_input), torch.FloatTensor(grad_filter)

    class ScipyConv2d(Module):
        def __init__(self, kh, kw):
            super(ScipyConv2d, self).__init__()
            self.filter = Parameter(torch.randn(kh, kw))

        def forward(self, input):
            return ScipyConv2dFunction()(input, self.filter)

    module = ScipyConv2d(3, 3)
    print(list(module.parameters()))
    input = torch.autograd.Variable(torch.randn(10, 10), requires_grad=True)
    output = module(input)
    print(output)
    output.backward(torch.randn(8, 8))
    print(input.grad)


if __name__ == "__main__":
    # test()
    # print("test done")

    x = q.var(torch.ones(4, 3), requires_grad=True).v
    a = q.var(torch.ones(4, 3), requires_grad=False).v

    tfm = DummyTFModule2()

    # TODO: at this point, tfm has no params (params only appear after calling underlying tensorflow domain apply)

    y, _ = tfm(x, a)
    optim = torch.optim.SGD(q.params_of(tfm), lr=1)
    # print(y)
    l = y.sum()
    print("doing backward")
    l.backward()

    print("torch-stored parameter value before update and flush:")
    # print(tfm._params[0])
    print("tensorflow-stored parameter value before update and flush:")
    # print(tfm._tfsess.run(tfm._param2tfparam[tfm._params[0]]))

    optim.step()

    print("torch-stored parameter value after update and before flush:")
    # print(tfm._params[0])
    print("tensorflow-stored parameter value after update and before flush:")
    # print(tfm._tfsess.run(tfm._param2tfparam[tfm._params[0]]))

    tfm.flush_params_to_tensorflow()

    print("torch-stored parameter value after update and flush:")
    # print(tfm._params[0])
    print("tensorflow-stored parameter value after update and flush:")
    # print(tfm._tfsess.run(tfm._param2tfparam[tfm._params[0]]))

    print("x grad")
    # print(x.grad)
    print("a grad")
    # print(a.grad)
    print("done")
    # print(tfm._params[0])
    # print(tfm._params[0].grad)