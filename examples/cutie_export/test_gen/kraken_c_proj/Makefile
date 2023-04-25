PULP_APP = dvs_cutie
PULP_APP_SRCS = main.c cutie_driver.c 
PULP_CFLAGS += -O3

ifdef gui
VSIM_RUNNER_FLAGS += -do ../cutie.tcl
endif

# assume that we are in a kraken-fe/regression_tests/<test_category>/<test_name> directory. If not, you need
# to specify the location of the padframe driver (or edit this makefile)

PADFRAME_LOC ?=../../../padframe/driver

ifdef io
ifeq ($(io),uart)
PULP_APP_FC_SRCS += $(PADFRAME_LOC)/src/kraken_padframe.c
PULP_CFLAGS += -I$(PADFRAME_LOC)/include
endif
endif

stackSize = 8192

include $(PULP_SDK_HOME)/install/rules/pulp.mk
