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
from ni_interface.ni_brian2 import *



set_device('cpp_standalone', build_on_run=False)
device = get_device()

# specify the location of the MNIST data
MNIST_data_path = './MNIST/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
################################################################################
# Helper functions that expose some internal Brian machinery
def initialize_parameter(variableview, value):
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name


def set_parameter_value(identifier, value):
    np.atleast_1d(value).tofile(os.path.join(device.project_dir,
                                             'static_arrays',
                                             identifier))

def run_again():
    device.run(device.project_dir, with_output=False, run_args=[])

def plot_state(state_monitor, spike_monitor):
    #plot I total and v
    spike_dict = spike_monitor.spike_trains()
    figure(figsize=(10, 4), num=0)
    clf()
    subplot(211)
    plot(state_monitor.t / ms, state_monitor.I_total[0] / nA)
    
    xlabel('Time (ms)')
    ylabel('I in (nA)')
    subplot(212)
    plot(state_monitor.t / ms, state_monitor.v[0] / mV)
    scatter(spike_dict[0] / ms, np.ones(len(spike_dict[0])) * -50, color='r')
    xlabel('Time (ms)')
    ylabel('v (mV)')
    pause(5)



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


def save_connections():
    print( 'save connections')
    conn = connections['XeAe']
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(data_path + 'weights/XeAe', connListSparse)

def save_theta():
    print( 'save theta')
    np.save(data_path + 'weights/theta_A', neuron_groups['Ae'].theta)

def normalize_weights(connections, n_e, n_i, connection_list):
    len_source = len(connections['XeAe'].source)
    len_target = len(connections['XeAe'].target)
    connection = np.zeros((len_source, len_target))
    connection[connection_list[:,0], connection_list[:,1]] = connections['XeAe'].w
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = 78./colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connections['XeAe'].w = temp_conn[connection_list[:,0], connection_list[:,1]]
    return connections

def run_network(j, rate, test_mode = False, state_prop=None):
    
    
    device.reinit()
    device.activate()
    set_device('cpp_standalone', build_on_run=False)
    

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
            I_total = I_synE + I_synI                                : amp
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

    neuron_groups = {}
    input_groups = {}
    connections = {}
    spike_counters = {}
    state_monitors = {}

    neuron_groups['Ae'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method='euler')
    neuron_groups['Ai'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, method='euler')

    dyn_clamp_neuron, group = attach_neuron(neuron_groups['Ae'], 0, 'v', 'I_total', dt=defaultclock.dt)
    dyn_clamp_neuron.thresh_offset_RT = 20*b.mV
    dyn_clamp_neuron.v = -65. * b.mV
    #------------------------------------------------------------------------------
    # create network population and recurrent connections
    #------------------------------------------------------------------------------
    print( 'create neuron group A')

    neuron_groups['Ae'].v = v_rest_e - 40. * b.mV
    neuron_groups['Ai'].v = v_rest_i - 40. * b.mV
    if test_mode:
        neuron_groups['Ae'].theta = np.load(data_path + 'weights/theta_A.npy') * b.volt
    elif not test_mode and state_prop is None:
        neuron_groups['Ae'].theta = np.ones((n_e)) * 20.0*b.mV
    else:
        neuron_groups['Ae'].theta = state_prop['theta'] * b.mV

    print( 'create recurrent connections')
    weightMatrix = get_matrix_from_file(data_path + 'random/AeAi.npy', n_e, n_i)
    connections['AeAi'] = b.Synapses(neuron_groups['Ae'], neuron_groups['Ai'], model='w : 1', on_pre='ge_post += w')
    Ai_idxs = np.arange(n_i)
    Ae_idxs = np.arange(n_e)
    #generate all to all conectivity
    AeAi_idxs = np.array(np.meshgrid(Ae_idxs, Ai_idxs)).T.reshape(-1,2)

    connections['AeAi'].connect(i=AeAi_idxs[:,0], j=AeAi_idxs[:,1]) # all-to-all connection
    connections['AeAi'].w = weightMatrix[AeAi_idxs[:,0], AeAi_idxs[:,1]]

    weightMatrix = get_matrix_from_file(data_path + 'random/AiAe.npy', n_i, n_e)
    connections['AiAe'] = b.Synapses(neuron_groups['Ai'], neuron_groups['Ae'], model='w : 1', on_pre='gi_post += w')
    #generate all to all conectivity
    AiAe_idxs = np.array(np.meshgrid(Ai_idxs, Ae_idxs)).T.reshape(-1,2)
    connections['AiAe'].connect(i=AiAe_idxs[:,0], j=AiAe_idxs[:,1]) # all-to-all connection
    connections['AiAe'].w = weightMatrix[AiAe_idxs[:,0], AiAe_idxs[:,1]]

    print( 'create monitors for Ae')
    spike_counters['Ae'] = b.SpikeMonitor(neuron_groups['Ae'])
    state_monitors['Ae'] = b.StateMonitor(neuron_groups['Ae'], ['v', 'ge', 'gi', 'I_total'], record=[0])
    #------------------------------------------------------------------------------
    # create input population and connections from input populations
    #------------------------------------------------------------------------------
    input_groups['Xe'] = b.PoissonGroup(n_input, 0*Hz)

    print( 'create connections between X and A')
    if test_mode:
        weightMatrix = get_matrix_from_file(data_path + 'weights/XeAe.npy', n_input, n_e)
    elif test_mode==False and state_prop is None:
        weightMatrix = get_matrix_from_file(data_path + 'random/XeAe.npy', n_input, n_e)
    elif test_mode==False and state_prop is not None:
        weightMatrix = state_prop['weights'].reshape(n_input, n_e)
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

    Xe_idxs = np.arange(n_input)
    Ae_idxs = np.arange(n_e)
    #generate all to all conectivity
    XeAe_idxs = np.array(np.meshgrid(Xe_idxs, Ae_idxs)).T.reshape(-1,2)
    connections['XeAe'].connect(i=XeAe_idxs[:,0], j=XeAe_idxs[:,1]) # all-to-all connection
    
    connections['XeAe'].w = weightMatrix[XeAe_idxs[:,0], XeAe_idxs[:,1]]

    if state_prop is not None:
        connections['XeAe'].delay = state_prop['delays'] * b.ms
    else:
        connections['XeAe'].delay = 'minDelay + rand() * deltaDelay'

    #normalize weights if test mode
    #if test_mode == False:
        #normalize_weights(connections, n_e=n_e, n_i=n_i, connection_list=XeAe_idxs)

    #------------------------------------------------------------------------------
    # run the simulation and set inputs
    #------------------------------------------------------------------------------

    net = Network()
    
    for obj_list in [neuron_groups, input_groups, connections, spike_counters, state_monitors]:
        for key in obj_list:
            net.add(obj_list[key])

    init_neuron_device(device, scalefactor_out=2)
    net.run(0*second)
    input_groups['Xe'].rates = rate * Hz
    net.run(single_example_time, report='text')
    input_groups['Xe'].rates = 0 * Hz
    net.run(resting_time)
    device.build(directory='output', compile=True, run=True, debug=False)
    state_prop = {'weights': np.copy(connections['XeAe'].w),'delays': np.copy(connections['XeAe'].delay /b.ms), 'theta': np.copy(neuron_groups['Ae'].theta / b.mV)}
    return spike_counters, state_monitors, state_prop


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
num_examples =  600 * 1 # 추가
ending    = '' # 추가
n_output  = 10 # 추가
n_e = 400
n_i = n_e
n_input = 784

input_intensity = 2.
start_input_intensity = input_intensity

previous_spike_count = np.zeros(n_e)
input_numbers = [0] * num_examples

if test_mode:
        result_monitor = np.zeros((num_examples,n_e))

state_prop=None
j = 0
while j < (int(num_examples)):
    if test_mode:
        rate = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        rate = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    spike_counters, state_monitors, state_prop = run_network(j, rate, test_mode, state_prop)
    plot_state(state_monitors['Ae'], spike_counters['Ae'])
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
    else:
        if test_mode:
            result_monitor[j,:] = current_spike_count
            input_numbers[j] = testing['y'][j%10000][0]
        if j % 100 == 0 and j > 0:
            print( 'runs done:', j, 'of', int(num_examples))
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