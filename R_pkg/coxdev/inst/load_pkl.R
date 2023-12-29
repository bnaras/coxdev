get_obj <- function(fname = '/Users/naras/GitHub/coxdev/R_pkg/coxdev/inst/dev100.pkl') {
  code  <- paste(
    c(
      "import pickle",
      paste0("junk = ", sprintf("open('%s', 'rb')", fname)),
      "junk2 = pickle.load(junk)",
      "junk.close()"
    ),
    collapse = "\n")
  reticulate::py_run_string(code)
  as.list(reticulate::py$junk2)
}

