#!/usr/bin/perl -w
use strict;
use File::Temp qw/ tempdir /;
my $prog = "reducer";

die "$prog <code file> [optional command]\n" if ($#ARGV < 0);
my $file = shift @ARGV;
die "$prog: [error] cannot read file $file\n" if (! -r $file);

# Create a backup of the fuke.
my $dir = tempdir( CLEANUP => 0 );
print "$prog: created temporary directory '$dir'\n";
my $srcFile = "$dir/$file";
`cp $file $srcFile`;

# Create the script.
my $scriptFile = "$dir/script";
open(OUT, ">$scriptFile") or die "$prog: cannot create '$scriptFile'\n";
my $reduceOut = "$dir/reduceOut";

my $command;
if (scalar(@ARGV) > 0) { $command = \@ARGV; }
else {
  my $compiler = "clang++-8";
  $command = [$compiler, "-fno-crash-diagnostics", "-x", "cuda", "-fcuda-rdc", "--cuda-path=/usr/local/cuda", "--cuda-gpu-arch=sm_60", "-std=c++11"];
}
push @$command, $srcFile;
my $commandStr = "@$command";

print OUT <<ENDTEXT;
#!/usr/bin/perl -w
use strict;
my \$BAD = 1;
my \$GOOD = 0;
`rm -f $reduceOut`;
my \$command = "$commandStr > $reduceOut 2>&1";
system(\$command);
open(IN, "$reduceOut") or exit(\$BAD);
my \$found = 0;
while(<IN>) {
  my \$line = \$_;
  if(\$line =~ /_ZNK4vtkm4exec22CellLocatorUniformGridINS_4cont20DeviceAdapterTagCudaEE8FindCellERKNS0_11FunctorBaseE/) {
    exit \$GOOD;
  }
  if(\$line =~ /_ZNK4vtkm14ArrayPortalRefINS_3VecIfLi3EEEE3GetEx/) {
    exit \$GOOD;
  }
}
exit \$BAD;
ENDTEXT
close(OUT);
`chmod +x $scriptFile`;

print "$prog: starting reduction\n";
sub multidelta {
    my $level = shift @_;
    system("multidelta -level=$level $scriptFile $srcFile");
}

for (my $i = 1 ; $i <= 1; $i++) {
  foreach my $level (0,4,5,8) {
    multidelta($level);
  }
}

# Copy the final file.
`cp $srcFile.bak $file.reduced`;
print "$prog: generated '$file.reduced";
