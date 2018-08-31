# Plots
base <- "/media/alyjak/7CA27419A273D65C/BrainAge_UKBB_2/outputs20180826/"
dat <- read.csv(file.path(base, "Averaged_Age.csv"))
outdir <- file.path(base, "BPAD_outdir")
dir.create(outdir, showWarnings = FALSE)

dat$Training <- as.factor(dat$Training)
dat$CN <- as.factor(dat$CN)
#dat$Testing <- as.factor(dat$Testing)

# library(devtools)
# install_github("kassambara/easyGgplot2")

# Plot of histogram of diff between actual and predicted brain age
library(easyGgplot2)


# BrainAge_Expanded_maxComb16_learningRate0.08_minZDiff1.1_maxDepth5	
# BrainAge_Expanded_maxComb16_learningRate0.12_minZDiff1.05_maxDepth5	
# BrainAge_Expanded_maxComb16_learningRate0.12_minZDiff1.1_maxDepth5

# BrainAge_Expanded_maxComb16_learningRate0.08_minZDiff1.05_maxDepth8

for (pname in names(dat)){
    if (!startsWith(pname, "BrainAge_maxComb")){
        next
   }
    print(paste0("Calculating BPAD for ", pname))
    ## Change the Predicted.Age pointer
    dat$Predicted.Age <- dat[[pname]]
    dat$BPAD <- dat$Predicted.Age - dat$Age
    
    
    Res_diff_plot = data.frame(Group = dat$Optimal_training,
                               Predicted_vs_Chronological = dat$BPAD)

    plt <- ggplot2.histogram(data=Res_diff_plot, xName='Predicted_vs_Chronological',
                      groupName='Group', binwidth = 2, brewerPalette="Spectral", alpha=0.5,
                      mainTitle = "",
                      addDensity=TRUE) + theme_minimal()
    # print(plt)

    # RMSE etc
    
    rmse <- function(error)
    {
      sqrt(mean(error^2))
    }
    
    # Function that returns Mean Absolute Error
    mae <- function(error)
    {
      mean(abs(error))
    }
    
    # Example data
    actual <- dat$Age
    predicted <- dat$Predicted.Age
    
    # Calculate error
    error <- actual - predicted
    
    # Example of invocation of functions
    cat(paste0("\tRMSE: ", rmse(error), "\n"))
    cat(paste0("\tmae: ", mae(error), "\n"))

    #save_res <- readline(prompt="Save? [y/n]")
    save_res = 'y'
    if (save_res == 'y'){
        pdf(file.path(outdir, paste0(pname, ".pdf")))
        print(plt)
        dev.off()
        zz = file(file.path(outdir, paste0(pname, '.txt')), open="wt")
        sink(file=zz)
        cat(paste0("RMSE: ", rmse(error), "\n"))
        cat(paste0("mae: ", mae(error), "\n"))
        sink()
        close(zz)
    }

    ## # Age_c_sq
    ## dat$age_sq <- dat$Age * dat$Age
    ## Age_c_sq <- scale(dat$age_sq, center = T, scale = T)
    ## dat$Age_c_sq <- Age_c_sq
    ## tester <- lm(dat$Average_Predicted_Age ~ dat$Age + dat$Age_c_sq)              
    ## summary(tester)              
#    readline(prompt="Continue? [enter]")
}
