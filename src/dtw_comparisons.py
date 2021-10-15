import numpy as np
import pandas as pd
from tqdm import tqdm

from dtw import accelerated_dtw
from tslearn.barycenters import dtw_barycenter_averaging


def calculate_pairwise_dtw(timeseries, output_df=True):
    '''Calculate DTW distance between all timeseries pairwise

    Parameters
    ----------
    timeseries : list of dict
        where key:value is {ID:str}:{timeseries:list}

    Returns
    -------
    list, list, list
        distances[i]: euclidian distance from `from_ts[i]` to `to_ts[i]`
    '''
    distance = []
    from_ts = []
    to_ts = []
    compared_log = []

    for record_left in tqdm(timeseries):
        id_ts1 = list(record_left.keys())[0]
        ts1 = record_left[id_ts1]

        for record_right in timeseries:
            id_ts2 = list(record_right.keys())[0]
            ts2 = record_right[id_ts2]

            # only compare diffferent ts
            if id_ts1 != id_ts2:
                # only compare non-empty ts
                if ts1 and ts2:
                    # only compare ts that were not compared
                    compared_pair = set([id_ts1, id_ts2])
                    if compared_pair not in compared_log:

                        # get measures
                        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(
                            x=np.array(ts1),
                            y=np.array(ts2),
                            dist='euclidean'
                        )

                        distance.append(d)
                        from_ts.append(id_ts1)
                        to_ts.append(id_ts2)
                        compared_log.append(compared_pair)

        if output_df:
            return pd.DataFrame(
                np.column_stack([distance, from_ts, to_ts]),
                columns=["distance", "from", "to"])
        else:
            return distance, from_ts, to_ts


def calculate_dtw_to_x(timeseries, x):
    '''Calculat distance of all timeseries to timeseries x

    Parameters
    ----------
    timeseries : list of dict
        where key:value is {ID:str}:{timeseries:list}
    x : list|np.array
        timeseries to compare with

    Returns
    -------
    list, list
        distances[i]: euclidian distance from `x` to `to_ts[i]`
    '''

    if isinstance(x, list):
        x = np.array(x)

    distances = []
    to_ts = []

    for record_right in timeseries:
        id_ts2 = list(record_right.keys())[0]
        ts2 = record_right[id_ts2]

        # only compare non-empty ts
        if ts2:
            # get measures
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(
                x=x,
                y=np.array(ts2),
                dist='euclidean'
            )

            distances.append(d)
            to_ts.append(id_ts2)

    return distances, to_ts


def calculate_dtw_groupavg(df, grouping_var, subgrouping_var, output_df=True, group_var=True):
    '''
    Calculate group average ts and 
    dtw distance from the group average to all timeseries in the group.

    Parameters
    ----------
    df : pd.DataFrame
        long df where timeseires is one column.
        grouping_var & subgrouping_var must be columns within df.
    grouping_var : str
        name of colum containing information about spliting groups
    subgrouping_var : str
        name of column containing information about spliting timeseries
    output_df : bool
        get results as a df instead? By default True

    Returns
    -------
    df 
        of three columns (distance, from, to)

    OR

    list, list, list
        distances[i]: euclidian distance from `from_ts[i]` to `to_ts[i]`
    '''
    distance = []
    from_ts = []
    to_ts = []
    group_ids = []

    for gr_name, gr_frame in tqdm(df.groupby(grouping_var)):
        group_ts = []
        # get per sub-group shape of avg timeseries
        for sub_gr_name, sub_gr_frame in gr_frame.groupby(subgrouping_var):
            ts = sub_gr_frame['value'].tolist()
            group_ts.append(ts)

        avg_ts = dtw_barycenter_averaging(group_ts)
        avg_ts_id = f'avg-{gr_name}'

        # calculate distance from group average to all ts in that group
        for sub_gr_name, sub_gr_frame in gr_frame.groupby(subgrouping_var):
            ts = sub_gr_frame['value'].tolist()
            # keep track of target node
            ts_id = set(sub_gr_frame['newspaper_event'])
            assert len(ts_id) == 1
            ts_id = list(ts_id)[0]

            # only compare different ts
            if ts_id != avg_ts_id: 
                d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(
                    x=np.array(avg_ts),
                    y=np.array(ts),
                    dist='euclidean'
                )

                distance.append(d)
                from_ts.append(avg_ts_id)
                to_ts.append(ts_id)
                group_ids.append(gr_name)

    if output_df:
        if group_var:
            return pd.DataFrame(
                np.column_stack([distance, from_ts, to_ts, group_ids]),
                columns=["distance", "from", "to", "group"])
        else:
            return pd.DataFrame(
                np.column_stack([distance, from_ts, to_ts]),
                columns=["distance", "from", "to"])

    else:
        if group_var:
            return distance, from_ts, to_ts, group_ids
        else:
            return distance, from_ts, to_ts
