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
from ni_interface.ni_brian2 import *
#load the classifier
import pickle
with open('svm_classifier.sav', 'rb') as f:
    clf = pickle.load(f)

defaultclock.dt = 0.1*ms

set_device('cpp_standalone', build_on_run=False)
#device = get_device()

# specify the location of the MNIST data
MNIST_data_path = '/home/smestern/brian2_SDTP/stdp-mnist-brian2/MNIST/'

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
        rates = digit * 256 * i * Hz
        #run the network for 350ms
        device.run(device.project_dir, run_args={input_groups_xe.rates: rates})

        #clear the input layer
        
        #get the output
        output = spike_counters_ae.count[:]
        i += 1
        device.run(device.project_dir, run_args={input_groups_xe.rates: np.full((784,), 0 * Hz)})
        if np.sum(output) > 0:
            break
    #feed it into the classifier
    out = clf.predict(output.reshape(1,-1))
    print("===== PREDICTION =====")
    print("Predicted: ", out)
    #print(output)

    

#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
DYN_CLAMP = True
test_mode = False

np.random.seed(0)
data_path = ''
'''
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3
'''
num_examples =  600 * 1 # 추가
ending    = '' # 추가
n_output  = 10 # 추가
n_e =40
n_i = n_e
n_input = 784

input_intensity = 2.
start_input_intensity = input_intensity

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples

if test_mode:
        result_monitor = np.zeros((num_examples,n_e))


single_example_time =   0.35 * b.second
resting_time = 0.25 * b.second
runtime = num_examples * (single_example_time + resting_time)

use_testing_set       = True # add

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = 'v=-45.*mV'
v_thresh_i = 'v>-40.*mV'
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms


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
v_thresh_e = '(v>(theta - offset + -52.*mV + thresh_offset_RT)) and (timer>refrac_e)'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        I_total = clip(I_synE + I_synI, -250*pA, 250*pA)                                : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        thresh_offset_RT : volt
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


neuron_groups_ae = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups_ai = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')
#------------------------------------------------------------------------------
# attach the dynamic clamp'd neuron to the network
#------------------------------------------------------------------------------
if DYN_CLAMP:
    dyn_clamp_neuron, group = attach_neuron(neuron_groups_ae, 0, 'v', 'I_total', dt=defaultclock.dt)
    dyn_clamp_neuron.thresh_offset_RT = 20*b.mV
    dyn_clamp_neuron.v = -65. * b.mV
#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
print( 'create neuron group A')

neuron_groups_ae.v = v_rest_e - 40. * b.mV
neuron_groups_ai.v = v_rest_i - 40. * b.mV
if test_mode:
    neuron_groups_ae.theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
else:
    neuron_groups_ae.theta = np.ones((n_e)) * 20.0*b.mV

print( 'create recurrent connections')
weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
connections_aeai = b.Synapses(neuron_groups_ae, neuron_groups_ai, model='w : 1', on_pre='ge_post += w')
#here we need to use the all-to-all connection, however we want to generate this using numpy. 
Ai_idxs = np.arange(n_i)
Ae_idxs = np.arange(n_e)
#generate all to all conectivity
AeAi_idxs = np.array(np.meshgrid(Ae_idxs, Ai_idxs)).T.reshape(-1,2)

connections_aeai.connect(i=AeAi_idxs[:,0], j=AeAi_idxs[:,1]) # all-to-all connection
connections_aeai.w = weightMatrix[AeAi_idxs[:,0], AeAi_idxs[:,1]]

weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
connections_aiae = b.Synapses(neuron_groups_ai, neuron_groups_ae, model='w : 1', on_pre='gi_post += w')
#generate all to all conectivity
AiAe_idxs = np.array(np.meshgrid(Ai_idxs, Ae_idxs)).T.reshape(-1,2)
connections_aiae.connect(i=AiAe_idxs[:,0], j=AiAe_idxs[:,1]) # all-to-all connection
connections_aiae.w = weightMatrix[AiAe_idxs[:,0], AiAe_idxs[:,1]]

print( 'create monitors for Ae')
spike_counters_ae = b.SpikeMonitor(neuron_groups_ae)
state_monitors_ae = b.StateMonitor(neuron_groups_ae, ['v', 'ge', 'theta', 'I_total'], record=[0,1])
#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
input_groups_xe = b.NeuronGroup(n_input, model='rates:Hz (constant)', threshold='rand()<rates*dt')
input_groups_xe.rates = np.full((n_input), 0.0, dtype=np.float32) * b.Hz

print( 'create connections between X and A')
if test_mode:
    weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
else:
    weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
model = 'w : 1'
pre = 'ge_post += w'
post = ''

if False: #not test_mode:
    print( 'create STDP for connection XeAe')
    model += eqs_stdp_ee
    pre += '; ' + eqs_stdp_pre_ee
    post = eqs_stdp_post_ee

connections_xeae = b.Synapses(input_groups_xe, neuron_groups_ae,
                                            model=model, on_pre=pre, on_post=post)
minDelay = 0*b.ms
maxDelay = 10*b.ms
deltaDelay = maxDelay - minDelay

Xe_idxs = np.arange(n_input)
Ae_idxs = np.arange(n_e)
#generate all to all conectivity
XeAe_idxs = np.array(np.meshgrid(Xe_idxs, Ae_idxs)).T.reshape(-1,2)
connections_xeae.connect(i=XeAe_idxs[:,0], j=XeAe_idxs[:,1]) # all-to-all connection
#we need to pregenerate the delays
delays = np.random.rand(n_input, n_e) * deltaDelay + minDelay

#connections_xeae.connect(True) # all-to-all connection
connections_xeae.delay = delays
connections_xeae.w = weightMatrix[XeAe_idxs[:,0], XeAe_idxs[:,1]]

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

# net = Network()
# for obj_list in [neuron_groups, input_groups, connections, spike_counters, state_monitors]:
#     for key in obj_list:
#         net.add(obj_list[key])
if DYN_CLAMP:
    device = init_neuron_device(device)
zero_arr = np.zeros(n_input)
#param_rate = initialize_parameter(input_groups_xe.rates, zero_arr )
run(single_example_time, report='text')

device.build(compile=True, run=False, debug=False)
#device.run(device.project_dir, with_output=False, run_args=[])




#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
start_time = time.time()

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
