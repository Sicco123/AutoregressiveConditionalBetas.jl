function GARCH_volatilities(x, GARCH_model = GARCH{1, 1})

    model = ARCHModels.fit(GARCH_model, x; meanspec=Intercept, dist=StdT) 

    volatilities_pred = volatilities(model) #predict(model, :volatility, 1)
    model

    intercept_value = means(model) #sum(x)/length(x) 

    return volatilities_pred, intercept_value[1] 
end