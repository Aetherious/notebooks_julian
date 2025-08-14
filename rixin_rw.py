"""
Read Athena++ output data files and fill in empty regions in each level.

Author: Rixin Li
Email: <rixin.li.astro@gmail.com>
"""

# Standard Library
import re
import warnings

# External Packages
import matplotlib.pyplot as plt
import numpy as np

# Local Imports
import athena_read as athr


def unique_with_atol(A, atol=1e-7):
    equals = np.isclose(A[:-1], A[1:], atol=atol)
    if equals[-1] is True:
        return (A[:-1])[~equals]
    else:
        return np.hstack([(A[:-1])[~equals], A[-1]])
    
def find_min_restrictable_level(max_level, block_size):
    if max_level == 0:
        return 0
    else:
        for _level in range(max_level-1, -1, -1):  # from 2nd finest to root level
            max_restrict_factor = 2**(max_level - _level)
            for _block_size in block_size:
                if (_block_size != 1 and _block_size % max_restrict_factor != 0):
                    return _level + 1
        return 0
    
def dust_var_check(key_name):
    pattern = r'dust_\d+_(vel[1-3]|mom[1-3]|rho|dens)'
    return bool(re.match(pattern, key_name))

def simplified_comp_check(key_name):
    return bool(re.match(r'[uvM]([1-3]|[xyzr]|phi|theta|[Ï†Ï•Î¸])', key_name))

def dust_simplified_check(key_name):
    return bool(re.match(r'rho(d|p)_\d+', key_name) or
                re.match(r'(w|W)([1-3]|[xyzr]|phi|theta|[Ï†Ï•Î¸])_\d', key_name)
                )

class DataMasking:
    
    def __init__(self, filename, q=None, **kwargs):
    
        ds_raw = athr.athdf(filename, raw=True, quantities=q, )  # quantities not used if raw=True
        self.ds_raw = ds_raw
        self.filename = filename
        self.kwargs = kwargs
        
        self.t = ds_raw['Time']
        self.var_names = ds_raw['VariableNames']
        self.ds_name = ds_raw['DatasetNames'][0].decode('utf-8')  # it is either b'cons' or b'prim'
        if q is None:  # todo: check if input quantities are included in VariableNames
            self.quantities = [x.decode('ascii', 'replace') for x in self.var_names]
        elif isinstance(q, str):
            self.quantities = [q, ]
        else:
            self.quantities = q
    
        # mostly copied from athena_read.py
        self.block_size = ds_raw['MeshBlockSize']
        self.root_grid_size = ds_raw['RootGridSize']
        self.levels = ds_raw['Levels']
        self.logical_locations = ds_raw['LogicalLocations']
        self.max_level = ds_raw['MaxLevel']  # it's level index, starting from 0
        self.old_num_mb = ds_raw['NumMeshBlocks']
        
        # for now, if ndim = 2, we assume only the first two dimensions are meaningful
        self.ndim = self.get_ndim()
        
        # w/o knowing the minimum level that the finest level can be restricted to, otherwise, we'll running into the following error:
        # "Block boundaries at finest level must be cell boundaries at desired level for subsampling or fast restriction to work" 
        # In other words, restrictions from the finest level to a certain coarse level will fail if:
        #   (A) a meshblock in the finest level is smaller than one cell in that coarse level;
        #   (B) a meshblock in the finest level only covers odd number of cells in that coarse level (even number is needed).
        # Thus, using np.log2 to determine min_restrict-able_level only works if Nx in meshblock is 2^(...); need to check (B) as well.
        #
        #block_refine_limit = np.log2(self.block_size)
        #self.block_refine_limit = int(np.floor(block_refine_limit[block_refine_limit>0].min()))
        self.min_restrictable_level = find_min_restrictable_level(self.max_level, self.block_size)
        
        # fill in void regions in coarse levels (N.B., since ds_raw ignore q-selection, blocks generated for void areas will only contain selected q)
        self.restrict_data()

        # more post-processing
        self.mb_min = np.vstack([ds_raw['x1f'].min(axis=1), ds_raw['x2f'].min(axis=1), ds_raw['x3f'].min(axis=1) ]).T
        self.mb_max = np.vstack([ds_raw['x1f'].max(axis=1), ds_raw['x2f'].max(axis=1), ds_raw['x3f'].max(axis=1) ]).T

        self.lev_bbox = np.zeros([self.max_level+1, 6])  # bounding box for each level
        for _lev in range(0, self.max_level+1):
            _cecs = [unique_with_atol(np.sort(ds_raw['x1f'][self.levels == _lev].flatten())), 
                     unique_with_atol(np.sort(ds_raw['x2f'][self.levels == _lev].flatten())), 
                     unique_with_atol(np.sort(ds_raw['x3f'][self.levels == _lev].flatten()))]
            self.lev_bbox[_lev] = [_cecs[0].min(), _cecs[0].max(), _cecs[1].min(), _cecs[1].max(), _cecs[2].min(), _cecs[2].max()]

        # define abbreviated data name for easier access
        self._simplified_scalars = {
            # for hydro: cons have dens, mom1/2/3, prim have rho, vel1/2/3
            # for dust fluid: similarly species j have dust_j_[rho|dens], dust_j_[vel|mom]1/2/3
            "dens": ["dens", "rho", ],
            "rho": ["dens", "rho", ], 
            "rhog": ["dens", "rho", ], 
            "P": ["press", ],
            "pres": ["press", ],
            "pressure": ["press", ],
            "E": ["Etot", ]
        }
        self._density_choices = {"cons": "dens", "prim": "rho"}
        self._mv_choices = {"cons": "mom", "prim": "vel"}
        ## from Athena++'s code, it is possible to have output contains [vel|mom] or [vel|mom]_xyz as vectors
        self._tensor_header = {"u": "vel",  "v": "vel",  "M": "mom"}

        # determine geometric dimension symbols
        self._xyz_order = {'1': '1', '2': '2', '3': '3'}
        if ds_raw['Coordinates'] == b"cartesian":
            self.coor_name = "Car"
            self._xyz_order.update({'x': '1', 'y': '2', 'z': '3'})
        elif ds_raw['Coordinates'] == b"cylindrical":
            self.coor_name = "Cyl"
            self._xyz_order.update({'r': '1', 'phi': '2', 'z': '3', 'Ï†': '2', 'Ï•': '2'})
        elif ds_raw['Coordinates'] == b"spherical_polar":
            self.coor_name = "Sph"
            self._xyz_order.update({'r': '1', 'theta': '2', 'phi': '3', 'Î¸': '2', 'Ï†': '3', 'Ï•': '3'})
        else:
            warnings.warn(f"`__getitem__` with simplified component names (e.g., 'ur') is currently not supported under this coordinates: {ds_raw['Coordinates']}")
            
        # finally, aggregate data at the lowest level by default     
        self.get_level(0)  # get_level now requires self.coor_name
        
    def __getitem__(self, _name):
        """ Overload operator [] """
        
        if _name in self.ds:
            return self.ds[_name].view()  # ds will only contain np.ndarray
        elif _name in self.ds_raw:
            return self.ds_raw[_name]  # could be other type
        elif _name in self._simplified_scalars:
            for item in self._simplified_scalars[_name]:
                if item in self.ds:
                    return self.ds[item].view()
                elif item in self.ds_raw:
                    return self.ds_raw[item]
            raise KeyError(f"{_name} not found.", self._simplified_scalars[_name], "not found as well.")
        elif simplified_comp_check(_name):
            real_wanted = self._tensor_header[_name[0]] + self._xyz_order[_name[1:]]
            try:
                return self.__getitem__(real_wanted)
            except:
                raise KeyError(f"{_name} not found. {real_wanted} not found as well.")
        elif dust_var_check(_name):  # this is not needed as it will be in ds.ds or ds.ds_raw
            raise RuntimeError("This branch should not happen.")
        elif dust_simplified_check(_name):
            if _name[0] == "r":
                # currently we only accept rhod_j or rhop_j, where j is the species index
                real_wanted = "dust_" + _name.split('_')[1] + "_" + self._density_choices[self.ds_name]
                try:
                    return self.__getitem__(real_wanted)
                except:
                    raise KeyError(f"{_name} not found. {real_wanted} not found as well.")
            elif _name[0] == "w" or _name[0] == "W":
                # currently we only accept w1/x/r/..._j, where j is the species index
                _comp, _species = _name.split('_')
                real_wanted = "dust_"+_species+"_"+self._mv_choices[self.ds_name]+self._xyz_order[_comp[1:]]
                try:
                    return self.__getitem__(real_wanted)
                except:
                    raise KeyError(f"{_name} not found. {real_wanted} not found as well.")                    
        else:
            raise KeyError(_name, " not found.")
    
    #def __setitem__(self, _name, _value):
    #    """ Overload assignment via operator [] """
    #    
    #    raise NotImplementedError("Writing access to data is not implemented.")
        
    def __contains__(self, _name):
        """ Overload operator `in` """
        
        if _name in self.ds:
            return True
        elif _name in self.ds_raw:
            return True
        else:
            return False

    def get_idx_mb(self, pos, lev=None):
        """ Get the MeshBlock index based on position """

        matched_idx_mbs = np.where(np.all((self.mb_min < pos) & (pos < self.mb_max), axis=1) == True)[0]
        if matched_idx_mbs.size == 0:  # all False
            raise ValueError("Seems the inquired position", pos, "is outside the domain.")
        if lev is None:
            idx_mb = matched_idx_mbs[np.argmax(self.levels[matched_idx_mbs])]
        else:
            if not np.any( self.levels[matched_idx_mbs] == lev ):
                print(f"Cannot find a MeshBlock on level {lev} for the inquired position", pos, "; using highest level")
                idx_mb = matched_idx_mbs[np.argmax(self.levels[matched_idx_mbs])]
            else:
                idx_mb = matched_idx_mbs[self.levels[matched_idx_mbs] == lev]
        return idx_mb

    def get_level(self, lev):
        """ Construct the entire mesh data for a certain level
            CAVEAT: assuming mesh in that level is contiguous cuboid (either with rectilinear or curvilinear edges)
                    note that mesh may not be contiguous cuboid with AMR
        """
        
        sel_mb_lev = np.where(self.levels == lev)[0]
        logi_locs = self.logical_locations[sel_mb_lev]
        anchor = logi_locs.min(axis=0)
        logi_locs -= anchor
        Nx_mb = self.block_size
        Nx_lev = Nx_mb * (logi_locs.max(axis=0) + 1)  # b/c locs starts from 0
        
        # reconstruct cell center coordinates
        ccx1, ccx2, ccx3 = np.zeros(Nx_lev[0]), np.zeros(Nx_lev[1]), np.zeros(Nx_lev[2])
        #cex1, cex2, cex3 = np.zeros(Nx_lev[0]+1), np.zeros(Nx_lev[1]+1), np.zeros(Nx_lev[2]+1)
        
        if self.ndim == 2:
            Nx_lev = (Nx_lev[:2])
            
        level_data = {}
        for _q in self.quantities:            
            level_data[_q] = np.zeros(Nx_lev[::-1], dtype=self.ds_raw[_q][0].dtype)
            if self.kwargs.get("AMRmask", False):
                level_data[_q] += np.nan  # a quick, dirty way to deal with irregular AMR level
        
        for idx_sel_mb, idx_mb in enumerate(sel_mb_lev):
            #print(idx_sel_mb, idx_mb)
            _ccx1, _ccx2, _ccx3 = self.ds_raw['x1v'][idx_mb], self.ds_raw['x2v'][idx_mb], self.ds_raw['x3v'][idx_mb]
            #_cex1, _cex2, _cex3 = self.ds_raw['x1f'][idx_mb], self.ds_raw['x2f'][idx_mb], self.ds_raw['x3f'][idx_mb]
            ccx1[Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _ccx1
            ccx2[Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1)] = _ccx2
            ccx3[Nx_mb[2]*logi_locs[idx_sel_mb][2]:Nx_mb[2]*(logi_locs[idx_sel_mb][2]+1)] = _ccx3
            
            for _q in self.quantities:
                if self.kwargs.get("same_level_mask", False) and idx_mb > self.old_num_mb:                    
                    _q_current_mb = np.nan
                else:
                    _q_current_mb = self.ds_raw[_q][idx_mb]
                if self.ndim == 2:
                    (level_data[_q])[Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1),
                                     Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _q_current_mb  #self.ds_raw[_q][idx_mb]
                        
                if self.ndim == 3:
                    (level_data[_q])[Nx_mb[2]*logi_locs[idx_sel_mb][2]:Nx_mb[2]*(logi_locs[idx_sel_mb][2]+1), 
                                     Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1),
                                     Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _q_current_mb  #self.ds_raw[_q][idx_mb]
                    
        for _q in self.quantities:
            level_data[_q] = np.ma.masked_invalid(level_data[_q])  # mask out missing data (will be transparent in colormap)
            if not level_data[_q].mask.any():    # Returns True if any of the elements of a evaluate to True (i.e., NaN exist)
                level_data[_q] = np.ma.getdata(level_data[_q])

        self._cecs = [unique_with_atol(np.sort(self.ds_raw['x1f'][self.levels == lev].flatten())), 
                      unique_with_atol(np.sort(self.ds_raw['x2f'][self.levels == lev].flatten())), 
                      unique_with_atol(np.sort(self.ds_raw['x3f'][self.levels == lev].flatten()))]
        
        # meaning: current "working level", its Nx, cell center coordinates, and level data set
        self.wlev, self.Nx_wlev, self.cccs, self.ds = lev, Nx_lev, [ccx1, ccx2, ccx3], level_data
        #return [ccx1, ccx2, ccx3], level_data  # uncomment this to turn on backward compatibility

        if self.coor_name == "Car":
            self.ccx, self.ccy, self.ccz = self.cccs
        elif self.coor_name == "Cyl":
            self.ccr, self.ccphi, self.ccz = self.cccs
        elif self.coor_name == "Sph":
            self.ccr, self.cctheta, self.ccphi = self.cccs
        else:
            pass
        
    def restrict_data(self):
        """ Restrict data to complete void areas in coarse levels
            first, from the finest level to 2nd finest level, and then toward coarser levels
        """
        
        for lev in range(self.max_level, 0, -1):
            
            logi_locs = self.logical_locations[self.levels==lev]
            logi_locs_parent = logi_locs // 2  # get their logi_locs at -1 level

            # to find and group fine mesh blocks that can be merged into one coarse mesh blocks
            unq, count = np.unique(logi_locs_parent, axis=0, return_counts=True)
            repeated_groups = unq[count>1]

            re_levels = []
            re_logi_locs = []
            re_data = {'x1f': [], 'x1v': [], 'x2f': [], 'x2v': [], 'x3f': [], 'x3v': [], }
            for _q in self.quantities:
                re_data[_q] = []

            for repeated_group in repeated_groups:
                repeated_idx = np.argwhere(np.all(logi_locs_parent == repeated_group, axis=1))
                #print(repeated_idx.ravel()) # one can check this so we know it is 2D or 3D

                # hard-coded for 3D (but seems to also work in 2D so far)
                idx_to_merge = np.argwhere(self.levels==lev)[repeated_idx.ravel()].ravel()

                # athr.athdf uses face coordinates to find the enclosure boundaries, so center-coordiantes are fine to capture the mesh blocks
                bounding_box = np.array([[self.ds_raw['x1v'][idx_to_merge].min(), self.ds_raw['x1v'][idx_to_merge].max()], 
                                         [self.ds_raw['x2v'][idx_to_merge].min(), self.ds_raw['x2v'][idx_to_merge].max()], 
                                         [self.ds_raw['x3v'][idx_to_merge].min(), self.ds_raw['x3v'][idx_to_merge].max()], 
                                        ])
                #if lev-1 < self.min_restrictable_level:  # identify levels that cannot be restricted by the finest level
                # or, we can speed up by only using +1 level to restrict new blocks
                if True:
                    update_pack = {'levels':self.levels, 'logical_locations':self.logical_locations, }
                    for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                        update_pack[_coord] = self.ds_raw[_coord]
                    for _q in self.quantities:
                        update_pack[_q] = self.ds_raw[_q] 
                else:
                    update_pack = None
                    
                _ds = athr.athdf(self.filename, level=lev-1, 
                                 fast_restrict=self.kwargs.get("fast_restrict", True), 
                                 subsample=self.kwargs.get("subsample", False), 
                                 quantities=self.quantities, 
                                 update_pack=update_pack,
                                 x1_min=bounding_box[0][0], x1_max=bounding_box[0][1], 
                                 x2_min=bounding_box[1][0], x2_max=bounding_box[1][1], 
                                 x3_min=bounding_box[2][0], x3_max=bounding_box[2][1], )

                re_levels.append(lev-1)
                re_logi_locs.append(repeated_group)
                for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                    re_data[_coord].append(_ds[_coord])
                for _q in self.quantities:
                    re_data[_q].append(_ds[_q])

            self.levels = np.hstack([self.levels, np.atleast_1d(re_levels)])
            self.logical_locations = np.vstack([self.logical_locations, np.array(re_logi_locs)])
            for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                self.ds_raw[_coord] = np.vstack([self.ds_raw[_coord], np.array(re_data[_coord])])
            for _q in self.quantities:
                self.ds_raw[_q] = np.vstack([self.ds_raw[_q], np.array(re_data[_q])])
                
        self.ds_raw['Levels'], self.ds_raw['LogicalLocations'] = self.levels, self.logical_locations
        self.ds_raw['NumMeshBlocks'] = self.logical_locations.shape[0]
        
    def get_ndim(self, num_ghost = 0):
        
        # mostly copied from athena_read.py
        nx_vals = []
        for d in range(3):
            if self.block_size[d] == 1 and self.root_grid_size[d] > 1:  # sum or slice
                other_locations = [location
                                   for location in zip(self.levels,
                                                       self.logical_locations[:, (d+1) % 3],
                                                       self.logical_locations[:, (d+2) % 3])]
                if len(set(other_locations)) == len(other_locations):  # effective slice
                    nx_vals.append(1)
                else:  # nontrivial sum
                    level = self.max_level
                    num_blocks_this_dim = 0
                    for level_this_dim, loc_this_dim in zip(self.levels,
                                                            self.logical_locations[:, d]):
                        if level_this_dim <= level:
                            possible_max = (loc_this_dim+1) * 2**(level-level_this_dim)
                            num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                        else:
                            possible_max = (loc_this_dim+1) // 2**(level_this_dim-level)
                            num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                    nx_vals.append(num_blocks_this_dim)
            elif self.block_size[d] == 1:  # singleton dimension
                nx_vals.append(1)
            else:  # normal case
                nx_vals.append(self.root_grid_size[d] * 2**self.max_level + 2 * num_ghost)
        nx1 = nx_vals[0]
        nx2 = nx_vals[1]
        nx3 = nx_vals[2]
        lx1 = nx1 // self.block_size[0]
        lx2 = nx2 // self.block_size[1]
        lx3 = nx3 // self.block_size[2]
        num_extended_dims = 0
        for nx in nx_vals:
            if nx > 1:
                num_extended_dims += 1
                
        return num_extended_dims
    
    def plot_levels(self, ax=None, figsize=(10, 10), proj='z+'):

        if ax is None:
            new_ax_flag = True
            fig, ax = plt.subplots(figsize=figsize)
        
        if proj == 'z+':  # z+ means a view at top toward bottom, vertically
            for lev in range(0, self.max_level+1):
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][1]], [self.lev_bbox[lev][2], self.lev_bbox[lev][2]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][1]], [self.lev_bbox[lev][3], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][0]], [self.lev_bbox[lev][2], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][1], self.lev_bbox[lev][1]], [self.lev_bbox[lev][2], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
        else:
            raise NotImplementedError("Other perspective not yet implemented.")

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax
        
    def plot_mbs(self, ax=None, figsize=(10, 10), proj='z+'):

        if ax is None:
            new_ax_flag = True
            fig, ax = plt.subplots(figsize=figsize)
        
        if proj == 'z+':  # z+ means a view at top toward bottom, vertically
            for idx_mb in range(self.old_num_mb):
                _bbox = np.vstack([self.mb_min[idx_mb], self.mb_max[idx_mb]]).T.flatten()
                ax.plot([_bbox[0], _bbox[1]], [_bbox[2], _bbox[2]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[0], _bbox[1]], [_bbox[3], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[0], _bbox[0]], [_bbox[2], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[1], _bbox[1]], [_bbox[2], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
        else:
            raise NotImplementedError("Other perspective not yet implemented.")

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax

if __name__ == "__main__":
    pass
