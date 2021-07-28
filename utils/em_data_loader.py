import json
import mne
import glob
import pandas

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

        # Load all the EEG data as dictionaries
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


def generate_participants_datasets(group_path, group_folders, group):
    for participant_id in group_folders:
        print("Processing participant " + participant_id)
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
        #add_em_events_melo_raw(file_p1_path, events_p1)
        #add_em_events_melo_raw(file_p2_path, events_p2)
        print(participant_id, "...generated events!")

        file_p1.close()
        file_p2.close()
        file_metadata.close()

        annotations = extract_annotations(group_path, participant_id)

        events = [e[0] for e in events_p1]
        events_id = [e[2] for e in events_p1]
        data_cut_p1 = split_dataset_p1(data_p1, group, annotations, events, events_id, samp_rate=250)

        print(participant_id, "...split data part 1!")

        events = [e[0] for e in events_p2]
        events_id = [e[2] for e in events_p2]
        data_cut_p2 = split_dataset_p1(data_p2, group, annotations, events, events_id, samp_rate=250)

        print(participant_id, "...split data part 2!")

        data_cut = data_cut_p1
        data_cut.update(data_cut_p2)
        data_cut['events_p1'] = events_p1
        data_cut['events_p2'] = events_p2

        out_file = open(group_path + '/' + participant_id + '/' + participant_id + "_prepared.json", "w")

        json.dump(data_cut, out_file)

        out_file.close()
        print(participant_id +
              "...merged all the data and saved in " +
              group_path + '/' +
              participant_id + '/' +
              participant_id +
              "_prepared.json" + " !")
        # participant['group'] = group
        # participant['data_p1'] = data_p1
        # participant['data_p2'] = data_p2
        # participant['playlist_p1'] = playlist_p1
        # participant['playlist_p2'] = playlist_p2
        # participant['p1_timestamp'] = p1_timestamp
        # participant['p2_timestamp'] = p2_timestamp
        # participants[participant_id] = participant



def split_dataset_p1(data_p1, group, annotations, events, events_id, samp_rate=250):
    sr = samp_rate  # Sampling Rate
    data = {}
    trial = 1

    # Cut the Resting State EO
    enter_rest_state_eo = [[], []]
    idx = events[0]
    enter_rest_state_eo[0] = data_p1['recording']['channelData'][0][idx: idx + (120 * sr)]
    enter_rest_state_eo[1] = data_p1['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[0]]] = {}
    data[event_labels[events_id[0]]]['eeg'] = enter_rest_state_eo

    # Cut the Resting State EC
    enter_rest_state_ec = [[], []]
    idx = events[1]
    enter_rest_state_ec[0] = data_p1['recording']['channelData'][0][idx: idx + (120 * sr)]
    enter_rest_state_ec[1] = data_p1['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[1]]] = {}
    data[event_labels[events_id[1]]]['eeg'] = enter_rest_state_ec

    # Cut the Trial 1 white noises and stimuli
    wn_t1_a = [[], []]
    idx = events[2]
    wn_t1_a[0] = data_p1['recording']['channelData'][0][idx + (15 * sr): idx + (30 * sr)]  # the first trial has 15 extra seconds of WN that we do not need
    wn_t1_a[1] = data_p1['recording']['channelData'][1][idx + (15 * sr): idx + (30 * sr)]
    data[event_labels[events_id[2]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[2]] + '_' + str(trial) + 'a']['eeg'] = wn_t1_a

    t1_a = [[], []]
    idx = events[3]
    t1_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[3]]] = {}
    data[event_labels[events_id[3]]]['eeg'] = t1_a
    data[event_labels[events_id[3]]]['annotations'] = annotations['trial_1a']


    wn_t1_b = [[], []]
    idx = events[4]
    wn_t1_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t1_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[4]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[4]] + '_' + str(trial) + 'b']['eeg'] = wn_t1_b
    trial += 1

    t1_b = [[], []]
    idx = events[5]
    t1_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[5]]] = {}
    data[event_labels[events_id[5]]]['eeg'] = t1_b
    data[event_labels[events_id[5]]]['annotations'] = annotations['trial_1b']

    # Cut the Trial 2 white noises and stimuli
    wn_t2_a = [[], []]
    idx = events[6]
    wn_t2_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[6]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[6]] + '_' + str(trial) + 'a']['eeg'] = wn_t2_a

    t2_a = [[], []]
    idx = events[7]
    t2_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[7]]] = {}
    data[event_labels[events_id[7]]]['eeg'] = t2_a
    data[event_labels[events_id[7]]]['annotations'] = annotations['trial_2a']


    wn_t2_b = [[], []]
    idx = events[8]
    wn_t2_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[8]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[8]] + '_' + str(trial) + 'b']['eeg'] = wn_t2_b
    trial += 1

    t2_b = [[], []]
    idx = events[9]
    t2_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[9]]] = {}
    data[event_labels[events_id[9]]]['eeg'] = t2_b
    data[event_labels[events_id[9]]]['annotations'] = annotations['trial_2b']


    # Cut the Trial 3 white noises and stimuli
    wn_t3_a = [[], []]
    idx = events[10]
    wn_t3_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[10]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[10]] + '_' + str(trial) + 'a']['eeg'] = wn_t3_a

    t3_a = [[], []]
    idx = events[11]
    t3_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[11]]] = {}
    data[event_labels[events_id[11]]]['eeg'] = t3_a
    data[event_labels[events_id[11]]]['annotations'] = annotations['trial_3a']


    wn_t3_b = [[], []]
    idx = events[12]
    wn_t3_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[12]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[12]] + '_' + str(trial) + 'b']['eeg'] = wn_t3_b
    trial += 1

    t3_b = [[], []]
    idx = events[13]
    t3_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[13]]] = {}
    data[event_labels[events_id[13]]]['eeg'] = t3_b
    data[event_labels[events_id[13]]]['annotations'] = annotations['trial_3b']


    # Cut the Trial 4 white noises and stimuli
    wn_t4_a = [[], []]
    idx = events[14]
    wn_t4_a[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_a[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[14]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[14]] + '_' + str(trial) + 'a']['eeg'] = wn_t4_a

    t4_a = [[], []]
    idx = events[15]
    t4_a[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_a[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[15]]] = {}
    data[event_labels[events_id[15]]]['eeg'] = t4_a
    data[event_labels[events_id[15]]]['annotations'] = annotations['trial_4a']


    wn_t4_b = [[], []]
    idx = events[16]
    wn_t4_b[0] = data_p1['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_b[1] = data_p1['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[16]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[16]] + '_' + str(trial) + 'b']['eeg'] = wn_t4_b

    t4_b = [[], []]
    idx = events[17]
    t4_b[0] = data_p1['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_b[1] = data_p1['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[17]]] = {}
    data[event_labels[events_id[17]]]['eeg'] = t4_b
    data[event_labels[events_id[5]]]['annotations'] = annotations['trial_4b']

    return data


def split_dataset_p2(data_p2, group, annotations, events, events_id, samp_rate=250):
    sr = samp_rate  # Sampling Rate
    data = {}
    trial = 1

    # Cut the Trial 1 white noises and stimuli
    wn_t1_a = [[], []]
    idx = events[0]
    wn_t1_a[0] = data_p2['recording']['channelData'][0][idx + (15 * sr): idx + (30 * sr)]  # the first trial has 15 extra seconds of WN that we do not need
    wn_t1_a[1] = data_p2['recording']['channelData'][1][idx + (15 * sr): idx + (30 * sr)]
    data[event_labels[events_id[0]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[0]] + '_' + str(trial) + 'a']['eeg'] = wn_t1_a

    t1_a = [[], []]
    idx = events[1]
    t1_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[1]]] = {}
    data[event_labels[events_id[1]]]['eeg'] = t1_a
    data[event_labels[events_id[1]]]['annotations'] = annotations['trial_1a']


    wn_t1_b = [[], []]
    idx = events[2]
    wn_t1_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t1_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[2]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[2]] + '_' + str(trial) + 'b']['eeg'] = wn_t1_b
    trial += 1

    t1_b = [[], []]
    idx = events[3]
    t1_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t1_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[3]]] = {}
    data[event_labels[events_id[3]]]['eeg'] = t1_b
    data[event_labels[events_id[3]]]['annotations'] = annotations['trial_1b']


    # Cut the Trial 2 white noises and stimuli
    wn_t2_a = [[], []]
    idx = events[4]
    wn_t2_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[4]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[4]] + '_' + str(trial) + 'a']['eeg'] = wn_t2_a

    t2_a = [[], []]
    idx = events[5]
    t2_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[5]]] = {}
    data[event_labels[events_id[5]]]['eeg'] = t2_a
    data[event_labels[events_id[5]]]['annotations'] = annotations['trial_2a']


    wn_t2_b = [[], []]
    idx = events[6]
    wn_t2_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t2_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[6]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[6]] + '_' + str(trial) + 'b']['eeg'] = wn_t2_b
    trial += 1

    t2_b = [[], []]
    idx = events[7]
    t2_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t2_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[7]]] = {}
    data[event_labels[events_id[7]]]['eeg'] = t2_b
    data[event_labels[events_id[7]]]['annotations'] = annotations['trial_2b']


    # Cut the Trial 3 white noises and stimuli
    wn_t3_a = [[], []]
    idx = events[8]
    wn_t3_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[8]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[8]] + '_' + str(trial) + 'a']['eeg'] = wn_t3_a

    t3_a = [[], []]
    idx = events[9]
    t3_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[9]]] = {}
    data[event_labels[events_id[9]]]['eeg'] = t3_a
    data[event_labels[events_id[9]]]['annotations'] = annotations['trial_3a']


    wn_t3_b = [[], []]
    idx = events[10]
    wn_t3_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t3_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[10]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[10]] + '_' + str(trial) + 'b']['eeg'] = wn_t3_b
    trial += 1

    t3_b = [[], []]
    idx = events[11]
    t3_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t3_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[11]]] = {}
    data[event_labels[events_id[11]]]['eeg'] = t3_b
    data[event_labels[events_id[11]]]['annotations'] = annotations['trial_3b']


    # Cut the Trial 4 white noises and stimuli
    wn_t4_a = [[], []]
    idx = events[12]
    wn_t4_a[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_a[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[12]] + '_' + str(trial) + 'a'] = {}
    data[event_labels[events_id[12]] + '_' + str(trial) + 'a']['eeg'] = wn_t4_a

    t4_a = [[], []]
    idx = events[13]
    t4_a[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_a[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[13]]] = {}
    data[event_labels[events_id[13]]]['eeg'] = t4_a
    data[event_labels[events_id[13]]]['annotations'] = annotations['trial_4a']


    wn_t4_b = [[], []]
    idx = events[14]
    wn_t4_b[0] = data_p2['recording']['channelData'][0][idx: idx + (15 * sr)]
    wn_t4_b[1] = data_p2['recording']['channelData'][1][idx: idx + (15 * sr)]
    data[event_labels[events_id[14]] + '_' + str(trial) + 'b'] = {}
    data[event_labels[events_id[14]] + '_' + str(trial) + 'b']['eeg'] = wn_t4_b

    t4_b = [[], []]
    idx = events[15]
    t4_b[0] = data_p2['recording']['channelData'][0][idx: idx + (60 * sr)]
    t4_b[1] = data_p2['recording']['channelData'][1][idx: idx + (60 * sr)]
    data[event_labels[events_id[15]]] = {}
    data[event_labels[events_id[15]]]['eeg'] = t4_b
    data[event_labels[events_id[15]]]['annotations'] = annotations['trial_4b']


    # Cut the Resting State EO
    exit_rest_state_eo = [[], []]
    idx = events[16]
    exit_rest_state_eo[0] = data_p2['recording']['channelData'][0][idx: idx + (120 * sr)]
    exit_rest_state_eo[1] = data_p2['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[16]]] = {}
    data[event_labels[events_id[16]]]['eeg'] = exit_rest_state_eo

    # Cut the Resting State EC
    exit_rest_state_ec = [[], []]
    idx = events[17]
    exit_rest_state_ec[0] = data_p2['recording']['channelData'][0][idx: idx + (120 * sr)]
    exit_rest_state_ec[1] = data_p2['recording']['channelData'][1][idx: idx + (120 * sr)]
    data[event_labels[events_id[17]]] = {}
    data[event_labels[events_id[17]]]['eeg'] = exit_rest_state_ec

    return data


def extract_annotations(group_path, participant_id):
    file_metadata_csv_path = glob.glob(group_path + '/' + participant_id + '/' + metadata_folder + '/' + '*.csv')[0]
    annotation_csv = pandas.read_csv(
        file_metadata_csv_path,
        encoding='utf-8')

    # To extract the annotations we need to get the array using Pandas to open the .csv file,
    # however Pandas will return a string containing the array. To work around this issue the
    # following code is used:
    # annotation_csv['mouse.x'][0][1:-1].split(',')
    # We have the 1:-1 here because we want to get rid of the square brackets that are in the string.
    # We then convert all the values to float using list comprehension:
    # [float(x) for x in annotation_csv['mouse.x'][0][1:-1].split(',')]

    annotations = {
        "trial_1a": {
            "x": [float(x) for x in annotation_csv['mouse.x'][0][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse.y'][0][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating.response"][0],
            "familiarity": annotation_csv["song_one_rating.response"][1]
        },
        "trial_1b": {
            "x": [float(x) for x in annotation_csv['mouse_2.x'][2][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_2.y'][2][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating.response"][2],
            "familiarity": annotation_csv["song_two_rating.response"][3]
        },
        "trial_2a": {
            "x": [float(x) for x in annotation_csv['mouse.x'][5][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse.y'][5][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating.response"][5],
            "familiarity": annotation_csv["song_one_rating.response"][6]
        },
        "trial_2b": {
            "x": [float(x) for x in annotation_csv['mouse_2.x'][7][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_2.y'][7][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating.response"][7],
            "familiarity": annotation_csv["song_two_rating.response"][8]
        },
        "trial_3a": {
            "x": [float(x) for x in annotation_csv['mouse.x'][10][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse.y'][10][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating.response"][10],
            "familiarity": annotation_csv["song_one_rating.response"][11]
        },
        "trial_3b": {
            "x": [float(x) for x in annotation_csv['mouse_2.x'][12][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_2.y'][12][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating.response"][12],
            "familiarity": annotation_csv["song_two_rating.response"][13]
        },
        "trial_4a": {
            "x": [float(x) for x in annotation_csv['mouse.x'][15][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse.y'][15][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating.response"][15],
            "familiarity": annotation_csv["song_one_rating.response"][16]
        },
        "trial_4b": {
            "x": [float(x) for x in annotation_csv['mouse_2.x'][17][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_2.y'][17][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating.response"][17],
            "familiarity": annotation_csv["song_two_rating.response"][18]
        },
        "trial_5a": {
            "x": [float(x) for x in annotation_csv['mouse_3.x'][20][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_3.y'][20][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating_2.response"][20],
            "familiarity": annotation_csv["song_one_rating_2.response"][21]
        },
        "trial_5b": {
            "x": [float(x) for x in annotation_csv['mouse_4.x'][22][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_4.y'][22][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating_2.response"][22],
            "familiarity": annotation_csv["song_two_rating_2.response"][23]
        },
        "trial_6a": {
            "x": [float(x) for x in annotation_csv['mouse_3.x'][25][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_3.y'][25][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating_2.response"][25],
            "familiarity": annotation_csv["song_one_rating_2.response"][26]
        },
        "trial_6b": {
            "x": [float(x) for x in annotation_csv['mouse_4.x'][27][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_4.y'][27][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating_2.response"][27],
            "familiarity": annotation_csv["song_two_rating_2.response"][28]
        },
        "trial_7a": {
            "x": [float(x) for x in annotation_csv['mouse_3.x'][30][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_3.y'][30][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating_2.response"][30],
            "familiarity": annotation_csv["song_one_rating_2.response"][31]
        },
        "trial_7b": {
            "x": [float(x) for x in annotation_csv['mouse_4.x'][32][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_4.y'][32][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating_2.response"][32],
            "familiarity": annotation_csv["song_two_rating_2.response"][33]
        },
        "trial_8a": {
            "x": [float(x) for x in annotation_csv['mouse_3.x'][35][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_3.y'][35][1:-1].split(',')],
            "liking": annotation_csv["song_one_rating_2.response"][35],
            "familiarity": annotation_csv["song_one_rating_2.response"][36]
        },
        "trial_8b": {
            "x": [float(x) for x in annotation_csv['mouse_4.x'][37][1:-1].split(',')],
            "y": [float(y) for y in annotation_csv['mouse_4.y'][37][1:-1].split(',')],
            "liking": annotation_csv["song_two_rating_2.response"][37],
            "familiarity": annotation_csv["song_two_rating_2.response"][38]
        }
    }
    # the json file where the output must be stored
    out_file = open(group_path + '/'
                    + participant_id + '/'
                    + participant_id
                    + "_annotations.json", "w")

    json.dump(annotations, out_file, indent=4)

    out_file.close()

    return annotations
