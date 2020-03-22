import sound_processing

samples = 'samples/men'
points = sound_processing.get_points_in_feature_space(samples)
sound_processing.plot_points(points)
