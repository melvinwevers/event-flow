from collections import ChainMap

import ndjson
import pandas as pd


def extract_timeseries(event_flow_window, window=None, remove_empty=True, undesired_newspapers=['all']):
    '''Extract timeseries from event flow files

    Parameters
    ----------
    event_flow_window : list of dict
        dataset after loading
    window : int, optional
        which window to extract? By default None (no curbing) 
        Turning on will result in timeseries now ranging from ts[i-window] to ts[i+window]
    remove_empty : bool, optional
        filter out empty timeseries from output? By default True
    undesired_newspapers : list, optional
        newspapers (lines/json objects) to not extract, by default ['all']

    Returns
    -------
    list of dict
        key: {newspaper}-{event}
        value: timeseries (list)
    '''
    timeseries = []
    for newspaper in event_flow_window:
        if newspaper['newspaper'] in undesired_newspapers:
            continue

        else:
            for key in newspaper:
                if key not in ['newspaper', 'window_size']:
                    id_ts = f'{newspaper["newspaper"]}-{key}'
                    ts = newspaper[key]

                    if window and remove_empty:
                        if ts:
                            current_window = int(len(ts)/2)
                            # can't upscale, only downscale
                            assert window < current_window
                            # center point of ts = ts[current_window]
                            ts = ts[current_window -
                                    window:current_window+window]

                    res = {id_ts: ts}

                    if remove_empty:
                        if ts:
                            timeseries.append(res)
                    else:
                        timeseries.append(res)

    return timeseries


def load_as_df(path):
    df = pd.read_json(path, lines=True)
    df = df.melt(id_vars=["newspaper", "window_size"],
                 var_name="event",
                 value_name="time_series")

    # remove empty (dangerous implementation)
    df = df[df.astype(str)['time_series'] != '[]']

    # remove very old events & aggregated newspapers
    df = df[df['newspaper'] != 'all']
    df = df[df['event'] != 'olympics_ams']
    df = df[df['event'] != 'mussolini']
    df = df[df['event'] != 'spanish_flu']
    df = df[df['event'] != 'vietnam_end']

    # de facto ID variable
    df['event_paper'] = df['event'] + '_' + df['newspaper']

    return df


def ts_to_frame(timeseries):
    one_dict = dict(ChainMap(*timeseries))
    # convert to long df
    df = pd.DataFrame(one_dict)
    df = df.melt(var_name='newspaper_event', value_name='value')

    news_event = [d for d in df['newspaper_event']]
    news_event_pair = [d.split('-') for d in news_event]

    df['newspaper'] = [d[0] for d in news_event_pair]
    df['event'] = [d[1] for d in news_event_pair]

    return df
