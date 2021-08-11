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
import getopt
import os
import sys
import scipy
import numpy as np
import matplotlib.cm as cmap



prefs.codegen.target = 'cython'

# specify the location of the MNIST data
MNIST_data_path = 'C:/MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
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


def save_connections():
    print( 'save connections')
    conn = connections['XeAe']
    connListSparse = zip(conn.i, conn.j, conn.w)
    np.save(data_path + 'weights/XeAe', connListSparse)

def save_theta():
    print( 'save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups['Ae'].theta)

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


#------------------------------------------------------------------------------
# Parse command line arguments
#------------------------------------------------------------------------------
stoc_enable = 1
opts, args = getopt.getopt(sys.argv[1:],"hs",["help", "stoc_enable"])

for opt,arg in opts:
  if opt in ("-h", "--help"):
     print ('---------------')
     print ('Usage Example:')
     print ('---------------')
     print (os.path.basename(__file__) + ' --help        -> Print script usage example')
     print (os.path.basename(__file__) + ' --stoc_enable -> Enable stochasticity')
     sys.exit(1)
  elif opt in ("-s", "--stoc_enable"):
     stoc_enable = 1

if(stoc_enable):
   print ('--------------------------------------------------------------------')
   print ('Synapses connecting the input and excitatory neurons are stochastic!')
   print ('--------------------------------------------------------------------')
else:
   print ('------------------------------------------------------------------------')
   print ('Synapses connecting the input and excitatory neurons are NOT stochastic!')
   print ('------------------------------------------------------------------------')

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
data_path = 'C:/MNIST/'
'''
if test_mode:
    num_examples = 10000
else:
    num_examples = 60000 * 3
'''
num_examples        = 6000 * 1 # 추가
ending    = '' # 추가
n_output  = 10 # 추가

n_input = 784
n_e = 400
n_i = n_e

single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second
runtime = num_examples * (single_example_time + resting_time)

use_testing_set       = True # add


p_switch = np.array([0.0909, 0.0935, 0.0961, 0.1014, 0.1041, 0.1080, 0.1133, 0.1199, \
                     0.1238, 0.1291, 0.1344, 0.1423, 0.1502, 0.1581, 0.1647, 0.1765, \
                     0.1779, 0.1950, 0.2069, 0.2187, 0.2266, 0.2411, 0.2556, 0.2819, \
                     0.2872, 0.3030, 0.3228, 0.3439, 0.3623, 0.3874, 0.4150, 0.4493, \
                     0.4888, 0.5191, 0.5455, 0.5626, 0.5863, 0.6087, 0.6337, 0.6469, \
                     0.6693, 0.6812, 0.6970, 0.7088, 0.7352, 0.7457, 0.7589, 0.7708, \
                     0.7800, 0.7866, 0.8011, 0.8103, 0.8169, 0.8274, 0.8340, 0.8393, \
                     0.8511, 0.8538, 0.8603, 0.8669, 0.8762, 0.8814, 0.8906])
p_switch = p_switch - p_switch[0]
i_switch = np.arange(1, 64)
i_scale  = 5
i_norm   = (i_switch*1.0) / i_scale


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



if test_mode:
    scr_e        = 'timer = 0*ms'
else:
    theta_plus_e = 1000  * b.mV
    theta_stop   = 100e3 * b.mV
    scr_e        = 'theta = theta+theta_plus_e; timer = 0*ms'

if(test_mode):
   v_thresh_e = '(interp(I_post, i_norm, p_switch) > rand())'
   #v_thresh_e = 'I_post>3'
else:
   v_thresh_e = '(interp(I_post, i_norm, p_switch) > rand()) * (img_label==post_label)'
   #v_thresh_e = '(I_post>3)* (img_label==post_label)'



neuron_eqs_e = '''
        I_post = (ge-gi)                            : 1
        I_synE = ge                                 : 1
        I_synI = gi                                 : 1
        dge/dt = -ge/(4.0*ms)                       : 1
        dgi/dt = -gi/(2.0*ms)                       : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  theta      : volt'
    neuron_eqs_e += '\n  img_label  : 1.0'
    neuron_eqs_e += '\n  post_label : 1.0'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        I_post = (ge-gi)      : 1
        I_synE = ge           : 1
        I_synI = gi           : 1
        dge/dt = -ge/(1.0*ms) : 1
        dgi/dt = -gi/(2.0*ms) : 1
        '''

eqs_stdp_ee = '''
            dpre/dt  = -pre/(tc_pre_ee)      : 1.0
            dpost/dt = -post/(tc_post_1_ee)  : 1.0
            '''
if(stoc_enable == 0):
   eqs_stdp_pre_ee  = 'pre += 1.'
   eqs_stdp_post_ee = 'w += (nu_ee_post * (pre - STDP_offset) * ((wmax_ee - w)**exp_ee_post)); post += 1.'
else:
   # hebb_dep_count   = np.zeros((n_input, n_e))
   # eqs_stdp_pre_ee  = 'w -= ((post>=STDP_offset_dep_neg)*1.0*(prob_dep_neg>rand('+str(n_e)+'))); pre = 1.; hebb_dep_count += ((post>=STDP_offset_dep_neg)*1.0)'
   eqs_stdp_pre_ee = 'w -= ((post>rand(' + str(n_e) + '))*1.0); pre = pre_rst'

   # hebb_pot_count   = np.zeros((n_input, n_e))
   # eqs_stdp_post_ee = 'rand_updt_onebit = rand('+str(n_input)+'); w += ((pre>=STDP_offset_pot)*1.0*(prob_pot>rand_updt_onebit)) + ((pre<=STDP_offset_dep)*-1.0*(prob_dep>rand_updt_onebit)); post = 1.; hebb_pot_count += ((pre>=STDP_offset_pot)*1.0); antihebb_dep_count += ((pre<=STDP_offset_dep)*1.0)'
   eqs_stdp_post_ee = 'w += ((pre>rand('+str(n_input)+'))*1.0); post = post_rst'

neuron_groups = {}
input_groups = {}
connections = {}
spike_counters = {}
if test_mode:
    result_monitor = np.zeros((num_examples,n_e))

neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')


#--------------------------------------------------------------------------
# SNN connectivity specification
#--------------------------------------------------------------------------
conn_structure         = 'sparse'
delay                  = {}
input_population_names = ['X']
population_names       = ['A']
input_connection_names = ['XA']
save_conns             = ['XeAe','AeAi', 'AiAe']
input_conn_names       = ['ee_input']
recurrent_conn_names   = ['ei', 'ie']
delay['ee_input']      = (0*b.ms,10*b.ms)
delay['ei_input']      = (0*b.ms,5*b.ms)



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
while j < (int(num_examples)):
    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = rate * Hz
    net.run(single_example_time, report='text')

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print( 'runs done:', j, 'of', int(num_examples))
        input_groups['Xe'].rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1


#------------------------------------------------------------------------------
# save results
#------------------------------------------------------------------------------
print( 'save results')
if not test_mode:
    save_theta()
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)