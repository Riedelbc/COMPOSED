base.path <- "/media/alyjak/7CA27419A273D65C/BrainAge_UKBB_2/outputs20180826/"
prefix <- "BrainAge"
subpaths <- list.files(path=base.path, pattern=prefix)
combined <- NULL
for (subdir in subpaths){
  files <- list.files(path=file.path(base.path, subdir), pattern="Repeat")
  df <- NULL
  for (idx in 1:length(files)){
      fn <- paste0("Repeat_", idx, "_predictions_per_subject.csv")
      dats <- read.csv(file.path(base.path,subdir,fn),header=TRUE, sep=",")
      if (!all(c("Subject.ID", "Predicted.Age", "Actual.Age") %in% names(dats))){
          print("Uh Oh, this repeat does not have the expected columns. Help!")
          browser()
      }
      dats$Actual.Age <- NULL
      if (is.null(df)){
          df <- dats
      } else {
          df <- merge(x=df, y=dats, by="Subject.ID", all=TRUE, suffixes=c('', idx))
      }
  }
  if (!"Subject.ID" == names(df)[1]){
      print("Uh oh, why isn't subject ID the first column?")
      browser()
  }
  if (is.null(combined)){
    combined <- data.frame(RID=df$Subject.ID)
  }
  combined[[subdir]] <- rowMeans(df[,2:ncol(df)])
}

## Get DX and actual age
pheno <- read.csv(file.path(base.path, '..', 'inputs', 'subj_pheno.csv'),
                  header=TRUE, sep=",")

combined <- merge(combined, pheno, by="RID")
write.csv(combined, file.path(base.path, "Averaged_Age.csv"), row.names = F)
