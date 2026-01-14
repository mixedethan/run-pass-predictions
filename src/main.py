import load
import preprocessing
import train_model
import config


def main():
   
    print()
    print("=" * 60)
    print("NFL Play Prediction Pipeline")
    print("=" * 60)
    

    print("\n[1/3] Loading data...")
    df = load.load_data()
    

    print("\n[2/3] Preprocessing data...")
    df_processed = preprocessing.preprocess_data(df, garbage_time_method=config.GARBAGE_TIME_METHOD)
    
    
    print("\n[3/3] Training models...")
    results = train_model.train_pipeline(df_processed, test_season=config.TEST_SEASON, save_models=True)
    
    # evaluate the models
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nDecision Model Metrics (Kick or Go):")
    print(f"  Accuracy: {results['decision_metrics']['accuracy']:.3f}")
    print(f"  F1 Score: {results['decision_metrics']['f1']:.3f}")
    print(f"  Precision Score: {results['decision_metrics']['precision']:.3f}")
    print(f"  Recall Score: {results['decision_metrics']['recall']:.3f}")


    print(f"\nPlay Model Metrics (Run or Pass):")
    print(f"  Accuracy: {results['play_metrics']['accuracy']:.3f}")
    print(f"  F1 Score: {results['play_metrics']['f1']:.3f}")
    print(f"  Precision Score: {results['play_metrics']['precision']:.3f}")
    print(f"  Recall Score: {results['play_metrics']['recall']:.3f}")
    print(f"  Precision Score: {results['play_metrics']['precision']:.3f}")
    print(f"  Baseline Score: {results['play_metrics']['baseline_accuracy']:.3f}")


if __name__ == "__main__":
    main()

