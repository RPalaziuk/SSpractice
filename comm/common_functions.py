import json
import logging
import traceback

import re
from datetime import datetime

import google.cloud.bigquery as bq
import numpy as np
from google.cloud import storage
from pandas.io.json import json_normalize
from google.cloud import pubsub_v1

# Road MapMatch function
def map_match_gps(name_metric, dataframe_road, dataframe_gps, max_distance=30):
    """
    Vectorized projection/rejection based "baby map matching"
    to find the optimal segment assignment for every gps point.

    Arguments:    dfRoad    pandas dataframe describing the road. One row per
                            segment. The following columns must be present:

                            ['x1','y1','x2','y2']

                  dfGps     pandas dataframe of the GPS cloud. The following
                            columns must be present:

                            ['X','Y']

                  max_dist  Maximum perpendicular distance outside of which
                            points will be considered too far from a segment
                            to be assigned.

    Returns:      A numpy array of the segment assignment for every point.
                  If len(dfGps)=m, then the shape of the returned array
                  is (m,).

    Example use:  dfGps['seg_idx']=assignSegment(dfRoad,dfGps)

    """
    # square the distance tolerance
    distance_square = max_distance ** 2
    # TODO: remove "if and elif", rename in metrics into one view
    if name_metric == 'road_degradation_rolling_resistance':
        # convert needed columns to numpy
        points = dataframe_gps[['x', 'y']].values
        square_1 = dataframe_road[['X0', 'Y0']].values
        square_2 = dataframe_road[['X1', 'Y1']].values
    elif name_metric == 'tyre_temperature_pressure':
        points = dataframe_gps[['X', 'Y']].values
        square_1 = dataframe_road[['X0', 'Y0']].values
        square_2 = dataframe_road[['X1', 'Y1']].values
    elif name_metric == 'truck_bunching_anomalies':
        points = dataframe_gps[['x', 'y']].values
        square_1 = dataframe_road[['x0', 'y0']].values
        square_2 = dataframe_road[['x1', 'y1']].values
    elif name_metric == 'road_quality' or name_metric == 'truck_average_speed':
        points = dataframe_gps[['x', 'y']].values
        square_1 = dataframe_road[['xcoord', 'ycoord']].values
        square_2 = dataframe_road[['x1', 'y1']].values

    # there is no spoon
    expanded_points = np.expand_dims(points, 0)
    expanded_square_1 = np.expand_dims(square_1, 1)

    subtracted_points_and_square_1 = np.subtract(expanded_points, expanded_square_1)
    subtracted_squares = np.subtract(square_2, square_1)
    expanded_subtracted_squares = np.expand_dims(subtracted_squares, 1)
    multiplied_squares = subtracted_points_and_square_1 * expanded_subtracted_squares
    sum_axis_multiplied_squares = np.sum(multiplied_squares, axis=2)
    subtraction_square = square_2 - square_1

    norm_expanded_subtraction_square = np.linalg.norm(subtraction_square, axis=1)
    norm_expanded_subtraction_square = np.expand_dims(norm_expanded_subtraction_square, 1)

    projection = sum_axis_multiplied_squares / norm_expanded_subtraction_square
    transposed_projection = np.transpose(projection)
    transposed_norm_subtraction_square = np.transpose(norm_expanded_subtraction_square)
    transposed_valid_projection = (transposed_projection >= 0) & (transposed_projection <=
                                                                  transposed_norm_subtraction_square)
    projection_valid = np.transpose(transposed_valid_projection)
    subtraction_square_hat = np.transpose(np.transpose(subtracted_squares) /
                                          np.transpose(norm_expanded_subtraction_square))

    expanded_projection = np.expand_dims(projection, 1)
    expanded_projection = np.transpose(expanded_projection, [2, 0, 1])

    expanded_subtraction_square_hat = np.expand_dims(subtraction_square_hat, 0)
    rejection_vector = expanded_subtraction_square_hat * expanded_projection
    transposed_a = np.transpose(subtracted_points_and_square_1, [1, 0, 2])

    rejection_vector = transposed_a - rejection_vector
    rejection_mag_square = np.transpose(np.sum(np.square(rejection_vector), axis=2))
    rejection_valid = rejection_mag_square <= distance_square

    rejection_projection_valid_stack = np.stack([projection_valid, rejection_valid])
    rejection_projection_valid = np.all(rejection_projection_valid_stack, axis=0)
    rejection_projection_valid_reduce = np.any(rejection_projection_valid, axis=0)

    rejection_inf = rejection_mag_square
    rejection_inf[~rejection_projection_valid] = float('inf')
    transposed_rejection_inf = np.transpose(rejection_inf)

    seg_assignment = np.argmin(transposed_rejection_inf, axis=1)
    seg_assignment[~rejection_projection_valid_reduce] = -1
    return seg_assignment


def _read_query_from_file(base_path, template_name, properties_map):
    with open(base_path + '/' + template_name + '.sql', 'r') as file:
        return file.read().format_map(properties_map)


def _read_dataframe_from_template(client, base_path, template_name, properties_map):
    query = _read_query_from_file(base_path, template_name, properties_map)
    df = client.query(query).result().to_dataframe()
    if df.empty:
        raise ValueError('No ' + template_name + ' rows using following query params:' + json.dumps(properties_map))
    return df


def _execute_query_from_template(client, base_path, template_name, properties_map):
    query = _read_query_from_file(base_path, template_name, properties_map)
    client.query(query).result(timeout=300)


def _try_to_find_date_str(blob_name):
    return re.search('([0-9]{8})', blob_name)


def _extract_time_from_geojson(blob):
    return datetime.strptime(_try_to_find_date_str(blob.name).group(0), '%m%d%Y')


def _load_from_gcs_max_geojson(bucket, site, file_path=None):
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    if file_path:
        max_geojson_blob = bucket.blob(file_path)
    else:
        geojson_blobs = filter(lambda blob: _try_to_find_date_str(blob.name),
                               bucket.list_blobs(delimiter='/', prefix=site + '/'))

        max_geojson_blob = max(geojson_blobs, key=lambda blob: _extract_time_from_geojson(blob))
    return max_geojson_blob.name, json_normalize(json.loads(max_geojson_blob.download_as_string())['features'])


def apply_conditions(summary, options, average, target):
    conditions = [(summary[average] > summary[target] * 0.1) & (
            summary[average] < summary[target] * 0.90),
                  (summary[average] >= summary[target] * 0.90) & (
                          summary[average] <= summary[target] * 1.10),
                  (summary[average] > summary[target] * 1.10) & (
                          summary[average] <= summary[target] * 1.20),
                  (summary[average] > summary[target] * 1.20)]

    return np.select(conditions, options, default='OTHER')


def _add_timestamp_information(dataframe, start_time, end_time):
    dataframe['START_TIMESTAMP'] = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    dataframe['END_TIMESTAMP'] = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f')


def _add_road_network_meta(dataframe, road_network_file_name):
    dataframe['ROAD_NETWORK_META_ORIGIN'] = road_network_file_name


def process(calculate_metric,
            attributes,
            bq_client,
            sql_templates_path):
    logging.info('incoming attributes: %s', attributes)

    dataframe = calculate_metric(
        lambda query_template: _read_dataframe_from_template(bq_client, sql_templates_path, query_template, attributes))

    return _process_internal(dataframe, attributes)


def process_with_geojson(calculate_metric,
                         attributes,
                         bq_client,
                         sql_templates_path,
                         geojson_loader=lambda site, geojson_path:
                            _load_from_gcs_max_geojson('teck-aha-road-networks', site, geojson_path),
                         geojson_path=None):
    road_network_file_name, road_network_df = geojson_loader(attributes['site'], geojson_path)
    attributes['road_network_file_name'] = road_network_file_name

    dataframe = calculate_metric(
        lambda query_template: _read_dataframe_from_template(bq_client, sql_templates_path, query_template, attributes),
        road_network_df)
    _add_road_network_meta(dataframe, road_network_file_name)
    return _process_internal(dataframe, attributes)

def _process_internal(dataframe, attributes):
    logging.info('incoming attributes: %s', attributes)

    start_time = attributes['start_time']
    end_time = attributes['end_time']

    _add_timestamp_information(dataframe, start_time, end_time)

    if dataframe.empty:
        raise ValueError('no rows for metric for start_time: ' + start_time + ', end_time:' + end_time)

    return dataframe


def process_and_write_to_bq(process, calculate_metric, metric_name, event, sql_templates_path='sql_templates'):
    attributes = event['attributes']
    destination_table = attributes['destination_table']
    project_id = attributes['project_id']
    geojson_path = attributes.get('geojson_path')
    bq_client = bq.Client()
    if geojson_path:
        final_df = process(calculate_metric, attributes, bq_client, sql_templates_path, geojson_path=geojson_path)
    else:
        final_df = process(calculate_metric, attributes, bq_client, sql_templates_path)
    final_df["INSERTION_TIMESTAMP"] = datetime.utcnow()
    final_df.to_gbq(destination_table, project_id, if_exists='append')


def persist_data(metric_name, event, sql_templates_path='sql_templates'):
    attributes = event['attributes']
    bq_client = bq.Client()
    persist_metric(bq_client, sql_templates_path, attributes)
    send_notification_to_pub_sub(metric_name, attributes)


def send_notification_to_pub_sub(metric_name, attributes):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(attributes['project_id'], attributes['notification_topic'])
    message = {
        'metric_name': metric_name,
        'site': attributes['site']
    }
    data = json.dumps(message).encode('utf-8')
    publisher.publish(topic_path, data=data)


def persist_metric(bq_client, sql_templates_path, attributes):
    _execute_query_from_template(bq_client, sql_templates_path, 'persist_metric', attributes)


def run_with_exception_logging(process_func, calculate_metric, metric_type, request):
    """

    :param process_func: func
    :param calculate_metric: func
    :param metric_type: str
    :param request: (flask.Request): HTTP request object.
    :return:
    """
    try:
        event = request.get_json()
        operation_type = event['attributes'].get('type')
        if operation_type and operation_type == 'persist':
            persist_data(metric_type, event)
        else:
            process_and_write_to_bq(process_func, calculate_metric, metric_type, event)
    except Exception:
        msg = traceback.format_exc()
        logging.error('Failed to process event {}, with error: {}'.format(request.get_data(), msg))
        raise
