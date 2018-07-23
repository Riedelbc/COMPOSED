base.path <- "/media/alyjak/7CA27419A273D65C/BrainAge_Expanded_ADNI1_n_ADNI2_n_LifspanBrain_n_Oasis/outputs20180723_2/"
prefix <- "BrainAge_Expanded_maxComb"
subpaths <- list.files(path=base.path, pattern=prefix)
combined <- NULL
for (subdir in subpaths){
  files <- list.files(path=file.path(base.path, subdir), pattern="Repeat")
  repeat.predictions <- lapply(files, function(fn) {
    return(read.csv(file.path(base.path,subdir,fn),header=TRUE, sep=",")[,2])
  })
  df <- do.call(cbind, repeat.predictions)
  if (is.null(combined)){
    combined <- data.frame(Results=read.csv(file.path(base.path,subdir,files[1]), header=TRUE, sep=',')[,1])
    combined$RID <- combined[,1]
    combined[,1] <- NULL
  }
  combined[[subdir]] <- rowMeans(df)
}

## Get DX and actual age
pheno <- read.csv(file.path(base.path, '..', 'inputs', 'subj_pheno.csv'),
                  header=TRUE, sep=",")

combined <- merge(combined, pheno, by="RID")
write.csv(combined, file.path(base.path, "Averaged_Age.csv"), row.names = F)
