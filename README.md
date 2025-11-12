This is a Model-Data Coevolution (MDC) framework that iteratively evolves synthetic datasets to reflect real-world variability. 
Utilizing a Simulated Annealing (SA) controller to optimize augmentation parameters, guiding dataset evolution based on real-world validation feedback.
Experimentally validate a novel augmentation technique, Stochastic Feature Decoupling (SFD), where independent, stochastic noise is applied to both raw sensor features and their mathematically-derived scaled counterparts.
By intentionally breaking the mathematical link between raw and scaled features, the model was forced to learn a more robust representation, making it "street-smart" for the imperfections of the real world.
