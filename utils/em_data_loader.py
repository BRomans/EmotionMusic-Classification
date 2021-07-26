import json
import mne
import glob
import numpy as np

eeg_folder = 'eeg_raw'
metadata_folder = 'metadata'
physio_folder = 'physio_raw'

event_codes = {
    'resting_EO': 1,
    'resting_EC': 2,
    'white_noise': 3,
    'EO/class_1_A': 111,
    'EO/class_1_B': 112,
    'EO/class_2_A': 121,
    'EO/class_2_B': 122,
    'EO/class_3_A': 131,
    'EO/class_3_B': 132,
    'EO/class_4_A': 141,
    'EO/class_4_B': 142,
    'EC/class_1_A': 211,
    'EC/class_1_B': 212,
    'EC/class_2_A': 221,
    'EC/class_2_B': 222,
    'EC/class_3_A': 231,
    'EC/class_3_B': 232,
    'EC/class_4_A': 241,
    'EC/class_4_B': 242,

}

event_labels = {
    1: 'resting_EO',
    2: 'resting_EC',
    3: "white_noise",
    111: 'EO/class_1_A',
    112: 'EO/class_1_B',
    121: 'EO/class_2_A',
    122: 'EO/class_2_B',
    131: 'EO/class_3_A',
    132: 'EO/class_3_B',
    141: 'EO/class_4_A',
    142: 'EO/class_4_B',
    211: 'EC/class_1_A',
    212: 'EC/class_1_B',
    221: 'EC/class_2_A',
    222: 'EC/class_2_B',
    231: 'EC/class_3_A',
    232: 'EC/class_3_B',
    241: 'EC/class_4_A',
    242: 'EC/class_4_B'
}


def generate_em_mne_events(file, trials, part_one=True, part_two=False):
    raw_file = open(file)
    raw_json = json.load(raw_file)
    status_data = raw_json['recording']['statusData']
    i = 0
    t_idx = []
    while i < len(status_data):
        if status_data[i] == 1.0:
            # print("Second: " + str(i/250) + " - Index: " + str(i))
            t_idx.append(i)
            i += 250
        else:
            i += 1
    events = []
    if part_one:
        events.append([t_idx[1], 0, event_codes['resting_EO']])
        events.append([t_idx[2], 0, event_codes['resting_EC']])
        events.append([t_idx[3], 0, event_codes['white_noise']])
        events.append([t_idx[4], 0, event_codes[trials[0]]])
        events.append([t_idx[5], 0, event_codes['white_noise']])
        events.append([t_idx[6], 0, event_codes[trials[1]]])
        events.append([t_idx[7], 0, event_codes['white_noise']])
        events.append([t_idx[8], 0, event_codes[trials[2]]])
        events.append([t_idx[9], 0, event_codes['white_noise']])
        events.append([t_idx[10], 0, event_codes[trials[3]]])
        events.append([t_idx[11], 0, event_codes['white_noise']])
        events.append([t_idx[12], 0, event_codes[trials[4]]])
        events.append([t_idx[13], 0, event_codes['white_noise']])
        events.append([t_idx[14], 0, event_codes[trials[5]]])
        events.append([t_idx[15], 0, event_codes['white_noise']])
        events.append([t_idx[16], 0, event_codes[trials[6]]])
        events.append([t_idx[17], 0, event_codes['white_noise']])
        events.append([t_idx[18], 0, event_codes[trials[7]]])
    if part_two:
        events.append([t_idx[1], 0, event_codes['white_noise']])
        events.append([t_idx[2], 0, event_codes[trials[0]]])
        events.append([t_idx[3], 0, event_codes['white_noise']])
        events.append([t_idx[4], 0, event_codes[trials[1]]])
        events.append([t_idx[5], 0, event_codes['white_noise']])
        events.append([t_idx[6], 0, event_codes[trials[2]]])
        events.append([t_idx[7], 0, event_codes['white_noise']])
        events.append([t_idx[8], 0, event_codes[trials[3]]])
        events.append([t_idx[9], 0, event_codes['white_noise']])
        events.append([t_idx[10], 0, event_codes[trials[4]]])
        events.append([t_idx[11], 0, event_codes['white_noise']])
        events.append([t_idx[12], 0, event_codes[trials[5]]])
        events.append([t_idx[13], 0, event_codes['white_noise']])
        events.append([t_idx[14], 0, event_codes[trials[6]]])
        events.append([t_idx[15], 0, event_codes['white_noise']])
        events.append([t_idx[16], 0, event_codes[trials[7]]])
        events.append([t_idx[17], 0, event_codes['resting_EO']])
        events.append([t_idx[18], 0, event_codes['resting_EC']])
    evefile = file.split(".json")[0] + ".eve"
    mne.write_events(evefile, events)
    raw_file.close()
    return events


def add_em_events_melo_raw(file, events):
    raw_file = open(file)
    raw_json = json.load(raw_file)
    raw_json["events"] = events

    # We save the changes
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(raw_json, f)  # WARNING: indent makes it pretty but consumes way more space on disk!!
        print("Successfully saved events into ", file)
    raw_file.close()


def parse_playlist(playlist_raw, group):
    playlist = []
    con_a = ''
    con_b = ''
    for elem in playlist_raw:
        song_a = elem[0]
        song_b = elem[1]
        song_a = song_a.split('res\\playlist\\')[1].split('.ogg')[0]
        song_b = song_b.split('res\\playlist\\')[1].split('.ogg')[0]
        if group == '1EOEC':
            con_a = 'EO/'
            con_b = 'EC/'
        elif group == '2ECEO':
            con_a = 'EC/'
            con_b = 'EO/'
        playlist.append(con_a + song_a)
        playlist.append(con_b + song_b)
    return playlist


def generate_participants_events(group_path, group_folders, group):
    for participant_id in group_folders:
        # Find and open all the .json files containing EEG data
        file_p1_path = glob.glob(group_path + '/' + participant_id + '/' + eeg_folder + '/p1/' + '*.json')[0]
        file_p2_path = glob.glob(group_path + '/' + participant_id + '/' + eeg_folder + '/p2/' + '*.json')[0]
        file_metadata_path = glob.glob(group_path + '/' + participant_id + '/' + metadata_folder + '/' + '*.json')[0]
        file_p1 = open(file_p1_path)
        file_p2 = open(file_p2_path)
        file_metadata = open(file_metadata_path)

        # Load all the physiological data as numpy arrays
        # bvp = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'BVP.csv')
        # eda = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'EDA.csv')
        # hr = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'HR.csv')
        # temp = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'TEMP.csv')

        # Load all the EEG data as dictionaries
        data_p1 = json.load(file_p1)
        data_p2 = json.load(file_p2)
        metadata = json.load(file_metadata)

        # Parse the metadata
        playlist_p1_raw = metadata['playlist_p1']['val']
        playlist_p2_raw = metadata['playlist_p2']['val']
        p1_timestamp = metadata['playlist_p1']['timestamp']
        p2_timestamp = metadata['playlist_p2']['timestamp']
        playlist_p1 = parse_playlist(playlist_p1_raw, group)
        playlist_p2 = parse_playlist(playlist_p2_raw, group)
        print(participant_id, "...loaded!")

        events_p1 = generate_em_mne_events(file_p1_path, playlist_p1, part_one=True, part_two=False)
        events_p2 = generate_em_mne_events(file_p2_path, playlist_p2, part_one=False, part_two=True)
        add_em_events_melo_raw(file_p1_path, events_p1)
        add_em_events_melo_raw(file_p2_path, events_p2)
        print(participant_id, "...generated events!")

        file_p1.close()
        file_p2.close()
        file_metadata.close()

        # participant['group'] = group
        # participant['data_p1'] = data_p1
        # participant['data_p2'] = data_p2
        # participant['playlist_p1'] = playlist_p1
        # participant['playlist_p2'] = playlist_p2
        # participant['p1_timestamp'] = p1_timestamp
        # participant['p2_timestamp'] = p2_timestamp
        # participants[participant_id] = participant


def split_dataset_p1(data_p1, metadata, events, events_id, samp_rate=250):
    sr = samp_rate  # Sampling Rate
    data = {}
    trial = 1

    # Cut the Resting State EO
    enter_rest_state_eo = [[], []]
    idx = events[0]
    enter_rest_state_eo[0] = data_p1['recording']['channelData'][0][idx: idx + (120 * sr)]
    enter_rest_state_eo[1] = data_p1['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[0]]] = {}
    data[event_labels[events_id[0]]]['data'] = enter_rest_state_eo

    # Cut the Resting State EC
    enter_rest_state_ec = [[], []]
    idx = events[1]
    enter_rest_state_ec[0] = data_p1['recording']['channelData'][0][idx: idx + (120 * sr)]
    enter_rest_state_ec[1] = data_p1['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[1]]] = {}
    data[event_labels[events_id[1]]]['data'] = enter_rest_state_ec

    # Cut the Trial 1 white noises and stimuli
    wn_t1_a = [[], []]
    idx = events[2]
    wn_t1_a[0] = data_p1['recording']['channelData'][0][idx + (15 * sr): idx + (30 * sr)]  # the first trial has 15 extra seconds of WN that we do not need
    wn_t1_a[1] = data_p1['recording']['channelData'][1][idx + (15 * sr): idx + (30 * sr)]
    data[event_labels[events_id[2]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[2]] + '_' + str(trial) + 'a']['data'] = wn_t1_a

    t1_a = [[], []]
    idx = events[3]
    t1_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[3]]] = {}
    data[event_labels[events_id[3]]]['data'] = t1_a

    wn_t1_b = [[], []]
    idx = events[4]
    wn_t1_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t1_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[4]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[4]] + '_' + str(trial) + 'b']['data'] = wn_t1_b
    trial += 1

    t1_b = [[], []]
    idx = events[5]
    t1_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[5]]] = {}
    data[event_labels[events_id[5]]]['data'] = t1_b

    # Cut the Trial 2 white noises and stimuli
    wn_t2_a = [[], []]
    idx = events[6]
    wn_t2_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[6]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[6]] + '_' + str(trial) + 'a']['data'] = wn_t2_a

    t2_a = [[], []]
    idx = events[7]
    t2_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[7]]] = {}
    data[event_labels[events_id[7]]]['data'] = t2_a

    wn_t2_b = [[], []]
    idx = events[8]
    wn_t2_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[8]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[8]] + '_' + str(trial) + 'b']['data'] = wn_t2_b
    trial += 1

    t2_b = [[], []]
    idx = events[9]
    t2_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[9]]] = {}
    data[event_labels[events_id[9]]]['data'] = t2_b

    # Cut the Trial 3 white noises and stimuli
    wn_t3_a = [[], []]
    idx = events[10]
    wn_t3_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[10]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[10]] + '_' + str(trial) + 'a']['data'] = wn_t3_a

    t3_a = [[], []]
    idx = events[11]
    t3_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[11]]] = {}
    data[event_labels[events_id[11]]]['data'] = t3_a

    wn_t3_b = [[], []]
    idx = events[12]
    wn_t3_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[12]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[12]] + '_' + str(trial) + 'b']['data'] = wn_t3_b
    trial += 1

    t3_b = [[], []]
    idx = events[13]
    t3_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[13]]] = {}
    data[event_labels[events_id[13]]]['data'] = t3_b

    # Cut the Trial 4 white noises and stimuli
    wn_t4_a = [[], []]
    idx = events[14]
    wn_t4_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[14]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[14]] + '_' + str(trial) + 'a']['data'] = wn_t4_a

    t4_a = [[], []]
    idx = events[15]
    t4_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[15]]] = {}
    data[event_labels[events_id[15]]]['data'] = t4_a

    wn_t4_b = [[], []]
    idx = events[16]
    wn_t4_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[16]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[16]] + '_' + str(trial) + 'b']['data'] = wn_t4_b

    t4_b = [[], []]
    idx = events[17]
    t4_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[17]]] = {}
    data[event_labels[events_id[17]]]['data'] = t4_b

    return data

def split_dataset_p2(data_p2, metadata, events, events_id, samp_rate=250):
    sr = samp_rate  # Sampling Rate
    data = {}
    trial = 1

    # Cut the Trial 1 white noises and stimuli
    wn_t1_a = [[], []]
    idx = events[0]
    wn_t1_a[0] = data_p2['recording']['channelData'][0][idx + (15 * sr): idx + (30 * sr)]  # the first trial has 15 extra seconds of WN that we do not need
    wn_t1_a[1] = data_p2['recording']['channelData'][1][idx + (15 * sr): idx + (30 * sr)]
    data[event_labels[events_id[0]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[0]] + '_' + str(trial) + 'a']['data'] = wn_t1_a

    t1_a = [[], []]
    idx = events[1]
    t1_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[1]]] = {}
    data[event_labels[events_id[1]]]['data'] = t1_a

    wn_t1_b = [[], []]
    idx = events[2]
    wn_t1_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t1_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[2]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[2]] + '_' + str(trial) + 'b']['data'] = wn_t1_b
    trial += 1

    t1_b = [[], []]
    idx = events[3]
    t1_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[3]]] = {}
    data[event_labels[events_id[3]]]['data'] = t1_b

    # Cut the Trial 2 white noises and stimuli
    wn_t2_a = [[], []]
    idx = events[4]
    wn_t2_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[4]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[4]] + '_' + str(trial) + 'a']['data'] = wn_t2_a

    t2_a = [[], []]
    idx = events[5]
    t2_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[5]]] = {}
    data[event_labels[events_id[5]]]['data'] = t2_a

    wn_t2_b = [[], []]
    idx = events[6]
    wn_t2_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[6]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[6]] + '_' + str(trial) + 'b']['data'] = wn_t2_b
    trial += 1

    t2_b = [[], []]
    idx = events[7]
    t2_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[7]]] = {}
    data[event_labels[events_id[7]]]['data'] = t2_b

    # Cut the Trial 3 white noises and stimuli
    wn_t3_a = [[], []]
    idx = events[8]
    wn_t3_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[8]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[8]] + '_' + str(trial) + 'a']['data'] = wn_t3_a

    t3_a = [[], []]
    idx = events[9]
    t3_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[9]]] = {}
    data[event_labels[events_id[9]]]['data'] = t3_a

    wn_t3_b = [[], []]
    idx = events[10]
    wn_t3_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[10]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[10]] + '_' + str(trial) + 'b']['data'] = wn_t3_b
    trial += 1

    t3_b = [[], []]
    idx = events[11]
    t3_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[11]]] = {}
    data[event_labels[events_id[11]]]['data'] = t3_b

    # Cut the Trial 4 white noises and stimuli
    wn_t4_a = [[], []]
    idx = events[12]
    wn_t4_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[12]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[12]] + '_' + str(trial) + 'a']['data'] = wn_t4_a

    t4_a = [[], []]
    idx = events[13]
    t4_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[13]]] = {}
    data[event_labels[events_id[13]]]['data'] = t4_a

    wn_t4_b = [[], []]
    idx = events[14]
    wn_t4_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[14]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[14]] + '_' + str(trial) + 'b']['data'] = wn_t4_b

    t4_b = [[], []]
    idx = events[15]
    t4_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[15]]] = {}
    data[event_labels[events_id[15]]]['data'] = t4_b

    # Cut the Resting State EO
    exit_rest_state_eo = [[], []]
    idx = events[16]
    exit_rest_state_eo[0] = data_p2['recording']['channelData'][0][idx: idx + (120 * sr)]
    exit_rest_state_eo[1] = data_p2['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[16]]] = {}
    data[event_labels[events_id[16]]]['data'] = exit_rest_state_eo

    # Cut the Resting State EC
    exit_rest_state_ec = [[], []]
    idx = events[17]
    exit_rest_state_ec[0] = data_p2['recording']['channelData'][0][idx: idx + (120 * sr)]
    exit_rest_state_ec[1] = data_p2['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[17]]] = {}
    data[event_labels[events_id[17]]]['data'] = exit_rest_state_ec

    return data