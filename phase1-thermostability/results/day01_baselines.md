# Day 1 baselines (AA composition + log length)

```
      split     model eval  spearman   rmse   mae
mixed_split      mean  val       NaN 10.622 7.290
mixed_split      mean test       NaN 11.626 8.133
mixed_split ridge_aac  val     0.282  9.147 6.952
mixed_split ridge_aac test     0.321  9.623 7.493
mixed_split    rf_aac  val     0.527  6.964 4.990
mixed_split    rf_aac test     0.516  7.550 5.609
      human      mean  val       NaN  4.600 3.764
      human      mean test       NaN  4.910 4.052
      human ridge_aac  val     0.469  4.129 3.312
      human ridge_aac test     0.540  4.157 3.338
      human    rf_aac  val     0.488  4.008 3.168
      human    rf_aac test     0.600  3.925 3.109
 human_cell      mean  val       NaN  5.417 4.512
 human_cell      mean test       NaN  5.719 4.780
 human_cell ridge_aac  val     0.454  4.894 3.942
 human_cell ridge_aac test     0.545  4.837 3.920
 human_cell    rf_aac  val     0.487  4.708 3.784
 human_cell    rf_aac test     0.585  4.626 3.755
```
