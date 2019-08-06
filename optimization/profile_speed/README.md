# Major conclusion from this experiment
1. Line profiling result ```analysis.txt``` shows that 90% of the time
was spent on numpy.save(...). I would say it's IO-bound and not much to improve.
