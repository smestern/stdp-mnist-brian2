'''
Created on 15.12.2014

@author: Peter U. Diehl
'''


import numpy as np
import os.path
import pickle as pickle
import brian2 as b
from struct import unpack
from brian2 import *
import time
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
#load the classifier
import pickle
with open('svm_classifier.sav', 'rb') as f:
    clf = pickle.load(f)




prefs.codegen.target = 'cython'

# specify the location of the MNIST data
MNIST_data_path = 'C:/Users/SMest/source/repos/stdp-mnist-brian2/MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName, n_src, n_tgt):
    readout = np.load(fileName)
    print( readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr




def normalize_weights():
    len_source = len(connections['XeAe'].source)
    len_target = len(connections['XeAe'].target)
    connection = np.zeros((len_source, len_target))
    connection[connections['XeAe'].i, connections['XeAe'].j] = connections['XeAe'].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections['XeAe'].w = temp_conn[connections['XeAe'].i, connections['XeAe'].j]

def clear(arg):
    ax.clear()
    #draw some zeros
    ax.imshow(np.zeros((28,28)), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    fig.canvas.draw()

def onselect(verts):
    global previous_spike_count 
    bins = np.linspace(0, 28, 29)
    #its an 28x28 grid so we need to find the square that was selected
    verts = np.vstack(verts)
    #find the intersection of the selected path and the grid
    #this is a bit of a hack but it works
    #bin the verts into a grid
    digit, xedges, yedges = np.histogram2d(verts[:,0], verts[:,1], bins=bins)
    digit = np.transpose(digit)
    ax.clear()
    ax.imshow(digit, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    fig.canvas.draw()

    #feed the digit into the network
    #reshape the digit into a vector
    i = 1
    output = np.zeros((10,))
    
    while True:
        digit = digit.reshape(784)
        #normalize the digit
        digit = digit / np.max(digit)
        #set the input layer to the digit
        input_groups['Xe'].rates = digit * 256 * i * Hz
        #run the network for 350ms
        net.run(0.35 * b.second)
        #clear the input layer
        input_groups['Xe'].rates = 0 * Hz
        #run the network for 150ms
        net.run(0.15 * b.second)
        #get the output
        output = spike_counters['Ae'].count[:] - previous_spike_count
        i += 1
        previous_spike_count = np.copy(spike_counters['Ae'].count[:])
        if np.sum(output) > 0:
            break
    #feed it into the classifier
    out = clf.predict(output.reshape(1,-1))
    print(out)
    print(output)

    

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

np.random.seed(0)
data_path = ''
'''
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3
'''
num_examples =  300 * 1 # 추가
ending    = '' # 추가
n_output  =2 # 추가

n_input = 784
n_e = 10
n_i = n_e

single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)

use_testing_set       = True # add

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0

if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + -52.*mV)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

neuron_groups = {}
input_groups = {}
connections = {}
spike_counters = {}
if test_mode:
    result_monitor = np.zeros((num_examples,n_e))

neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print( 'create neuron group A')

neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
if test_mode:
    neuron_groups['Ae'].theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    neuron_groups['Ae'].theta = np.ones((n_e)) * 20.0*b.mV

print( 'create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
connections['AeAi'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model='w : 1', on_pre='ge_post += w')
connections['AeAi'].connect(True) # all-to-all connection
connections['AeAi'].w = weightMatrix[connections['AeAi'].i, connections['AeAi'].j]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
connections['AiAe'] = b.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model='w : 1', on_pre='gi_post += w')
connections['AiAe'].connect(True) # all-to-all connection
connections['AiAe'].w = weightMatrix[connections['AiAe'].i, connections['AiAe'].j]

print( 'create monitors for Ae')
spike_counters['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])

#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
input_groups['Xe'] = b.PoissonGroup(n_input, 0*Hz)

print( 'create connections between X and A')
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''

if not test_mode:
    print( 'create STDP for connection XeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee

connections['XeAe'] = b.Synapses(input_groups['Xe'], neuron_groups['Ae'],
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

connections['XeAe'].connect(True) # all-to-all connection
connections['XeAe'].delay = 'minDelay + rand() * deltaDelay'
connections['XeAe'].w = weightMatrix[connections['XeAe'].i, connections['XeAe'].j]


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples
input_groups['Xe'].rates = 0 * Hz
net.run(0*second)
j = 0
np.random.seed(42)
numbers_to_obs = np.arange(4).tolist()

plt.ion()
start_time = time.time()

fig, ax = plt.subplots(1, 1)
#plot an 8,8 image of zeros
imshowshow = ax.imshow(np.zeros((28,28)), cmap='gray')

#add a lasso selector
# line defines the color, width and opacity
# of the line to be drawn
line = {'color': 'white', 
        'linewidth': 8, 'alpha': 1}
lasso = LassoSelector(ax, onselect, lineprops=line)
#add a button to clear the selection
from matplotlib.widgets import Button
button = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Clear')
button.on_clicked(clear)

# try:
while True:
    
    fig.waitforbuttonpress()
#         #get the next input
#         input_numbers[j] = np.random.randint(0, 4)
#         input_numbers[j] = numbers_to_obs[j]
#         print( 'input number:', input_numbers[j])
#         #input_groups['Xe'].rates = input_images[input_numbers[j]] * Hz
#         #net.run(0.35*second)
#         input_groups['Xe'].rates = 0 * Hz
#         #net.run(0.15*second)    

#         #get the spikes
#         #spike_id = spike_counters['Ae'].count()
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass
print( 'simulation time:', time.time() - start_time)
