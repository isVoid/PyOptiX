import optix as ox
import cupy as cp

import array
import pytest


class TestProgramGroupOptions:
    def test_constructor(self):
        pgo = ox.ProgramGroupOptions()
        assert type(pgo) is ox.ProgramGroupOptions
        # ProgramGroupOptions is an empty data type. There is nothing to test here (yet).

class TestProgramGroupBase:
    def setup_method(self):
        self.ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        self.mod, log = self.ctx.moduleCreateFromPTX(ox.ModuleCompileOptions(),
                                                     ox.PipelineCompileOptions(),
                                                     tutil.ptx_string)

    def teardown_method(self):
        self.mod.destroy()
        self.ctx.destroy()


class TestProgramGroupDescriptor(TestProgramGroupBase):
    def test_constructor(self):
        pgd = ox.ProgramGroupDesc(raygenModule = self.mod,
                                  raygenEntryFunctionName = "__raygen__hello")
        assert pgd.raygenModule == self.mod
        assert pgd.raygenEntryFunctionName == "__raygen__hello"

    def test_attributes(self):
        pgd = ox.ProgramGroupDesc()
        pgd.raygenModule = self.mod
        pgd.raygenEntryFunctionName = "__raygen__hello"
        assert pgd.raygenModule == self.mod
        assert pgd.raygenEntryFunctionName == "__raygen__hello"


class TestProgramGroup(TestProgramGroupBase):
    def test_create_raygen(self):
        prog_group_opts = ox.ProgramGroupOptions()

        prog_group_desc                          = ox.ProgramGroupDesc()
        prog_group_desc.raygenModule             = self.mod
        prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

        prog_groups, log = self.ctx.programGroupCreate([prog_group_desc], prog_group_opts)
        assert len(prog_groups) == 1
        assert type(prog_groups[0]) is ox.ProgramGroup

        prog_groups[0].destroy()

    def test_create_miss(self):
        prog_group_opts = ox.ProgramGroupOptions()

        prog_group_desc                        = ox.ProgramGroupDesc()
        prog_group_desc.missModule             = self.mod
        prog_group_desc.missEntryFunctionName  = "__miss__noop"

        prog_groups, log = self.ctx.programGroupCreate([prog_group_desc], prog_group_opts)
        assert len(prog_groups) == 1
        assert type(prog_groups[0]) is ox.ProgramGroup

        prog_groups[0].destroy()

    def test_create_callables(self):
        prog_group_opts = ox.ProgramGroupOptions()

        prog_group_desc                               = ox.ProgramGroupDesc()
        prog_group_desc.callablesModuleDC             = self.mod
        prog_group_desc.callablesModuleCC             = self.mod
        prog_group_desc.callablesEntryFunctionNameCC  = "__continuation_callable__noop"
        prog_group_desc.callablesEntryFunctionNameDC  = "__direct_callable__noop"

        prog_groups, log = self.ctx.programGroupCreate([prog_group_desc], prog_group_opts)
        assert len(prog_groups) == 1
        assert type(prog_groups[0]) is ox.ProgramGroup

        prog_groups[0].destroy()

    def test_create_hitgroup(self):
        prog_group_opts = ox.ProgramGroupOptions()

        prog_group_desc                              = ox.ProgramGroupDesc()
        prog_group_desc.hitgroupModuleCH             = self.mod
        prog_group_desc.hitgroupModuleAH             = self.mod
        prog_group_desc.hitgroupModuleIS             = self.mod
        prog_group_desc.hitgroupEntryFunctionNameCH  = "__closesthit__noop"
        prog_group_desc.hitgroupEntryFunctionNameAH  = "__anyhit__noop"
        prog_group_desc.hitgroupEntryFunctionNameIS  = "__intersection__noop"

        prog_groups, log = self.ctx.programGroupCreate([prog_group_desc], prog_group_opts)
        assert len(prog_groups) == 1
        assert type(prog_groups[0]) is ox.ProgramGroup

        prog_groups[0].destroy()

    def create_prog_group(self):
        prog_group_opts = ox.ProgramGroupOptions()

        prog_group_desc                          = ox.ProgramGroupDesc()
        prog_group_desc.raygenModule             = self.mod
        prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

        prog_groups, log = self.ctx.programGroupCreate([prog_group_desc], prog_group_opts)
        return prog_groups[0]

    def test_get_stack_size(self):
        prog_group = self.create_prog_group()
        stack_size = prog_group.getStackSize()
        assert type(stack_size) is ox.StackSizes
