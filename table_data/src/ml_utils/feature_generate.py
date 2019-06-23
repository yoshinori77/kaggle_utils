import featuretools as ft


class FeatureGenerate(object):
    def __init__(self, data, cols, is_agg=True,
                 methods=['count', 'max', 'mean']):
        self.data = data
        self.cols = cols
        self.is_agg = is_agg
        self.methods = methods

    def execute(self):
        if self.is_agg:
            return self._featuretools_agg(self.methods)
        else:
            return self._featuretools_trans(self.methods)

    def _featuretools_agg(self, methods=['count', 'max', 'mean']):
        es = ft.EntitySet(id='index')
        es.entity_from_dataframe(entity_id='data',
                                 dataframe=self.data,
                                 index='index')
        for col in self.cols:
            es.normalize_entity(base_entity_id='data',
                                new_entity_id=col,
                                index=col
                                )
        features, _ = ft.dfs(entityset=es,
                             target_entity='data',
                             agg_primitives=methods,
                             max_depth=2,
                             verbose=1,
                             n_jobs=-1)
        return features

    def _featuretools_trans(self, methods=['multiply']):
        es = ft.EntitySet(id='index')
        es.entity_from_dataframe(entity_id='data',
                                 dataframe=self.data,
                                 index='index')
        features, _ = ft.dfs(entityset=es,
                             target_entity='data',
                             trans_primitives=methods,
                             max_depth=2,
                             verbose=1,
                             n_jobs=4)
        return features
