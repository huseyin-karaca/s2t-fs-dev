    def save_results(self, wers, margin, searches, output_filename=None):
        best_hyperparameters = {}
        for name, search in searches.items():
            best_idx = search.best_index_
            cv = search.cv_results_
            best_hyperparameters[name] = {
                "params": search.best_params_,
                "cv_mean_fit_time": float(cv["mean_fit_time"][best_idx]),
                "cv_mean_score_time": float(cv["mean_score_time"][best_idx]),
                "cv_mean_test_score": -float(cv["mean_test_score"][best_idx]),
                "cv_mean_train_score": (
                    -float(cv["mean_train_score"][best_idx])
                    if "mean_train_score" in cv else None
                ),
            }

        results = {
            "config": {
                "data_params": self.data_params,
                "search_params": self.search_params,
                "search_spaces": self.hyperparameter_spaces,
            },
            "results": {
                "dataset_stats": self.dataset_stats,
                "wers": wers,
                "margin": margin,
                "best_hyperparameters": best_hyperparameters,
            },
        }

        filename = output_filename or f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Results saved to %s", path)t