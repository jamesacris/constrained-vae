import tensorflow as tf
import multiprocessing

def simple_tf_matmul(vGPU, a_1):
    with tf.device(vGPU):
        a = tf.constant([[a_1,2],[2,4],[1,2]])
        b = tf.constant([[1,2,3],[2,4,6]])
        return tf.matmul(a, b)

if __name__ == "__main__":

    pGPUs = tf.config.list_physical_devices('GPU')
    vGPUs = []
    for pGPU in pGPUs:
        tf.config.experimental.set_virtual_device_configuration(
            pGPU,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        vGPUs.extend(tf.config.experimental.list_logical_devices('GPU'))

    # each GPU ==> 5 virtual vGPUs
    print(vGPUs)

    args_pool = [(vGPUs[0].name, 1), (vGPUs[1].name, 2)]
    with multiprocessing.Pool(processes=2) as pool:
        pool.starmap(simple_tf_matmul, args_pool)