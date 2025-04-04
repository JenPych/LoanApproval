# first line: 172
        @memory.cache
        def cached_grid_search_fit(pipeline, param_grid, X_train, y_train):
            grid_search = GridSearchCV(pipeline, param_grid, n_jobs= -1, cv= 3, scoring= 'f1')
            grid_search.fit(X_train, y_train)
            return grid_search
